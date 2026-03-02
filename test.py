import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import anthropic
from groq import Groq
from dotenv import load_dotenv
import base64
import json
import io
from pathlib import Path
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

load_dotenv()

# Load API keys from Streamlit secrets (cloud) or .env (local)
import os
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

# -----------------------------------------
# STYLING & PAGE SETUP
# -----------------------------------------
st.set_page_config(page_title="Aziz Delicatesse | Competitor Price Intelligence", layout="wide", page_icon="🍷")

# -----------------------------------------
# LOGIN
# -----------------------------------------
VALID_USERNAME = "admin"
VALID_PASSWORD = "admin"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    logo_login = Path(__file__).parent / "data" / "aziz-artguru.png"
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.image(str(logo_login), width=220)
        st.markdown("### Welcome to Aziz Delicatesse")
        st.markdown("Please log in to continue.")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.stop()

st.markdown("""
    <style>
    .stApp { background-color: #FCFCFC; }
    h1, h2, h3 { color: #1c1c1c; font-family: 'Georgia', serif; }
    .stDataFrame { border: 1px solid #eaeaea; border-radius: 5px; }
    .css-1d391kg { background-color: #f4f4f4; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# AI ANALYSIS FUNCTION
# -----------------------------------------
def analyze_menu_image(image, filename="Unknown"):
    """Sends the image to Claude vision model to extract restaurant name, items and prices."""
    try:
        # Resize to max 1024px on longest side before encoding
        image = image.convert("RGB")
        image.thumbnail((1024, 1024), Image.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        prompt = """
        You are an expert data extractor. Look at this restaurant menu image carefully.

        STEP 1: Find the restaurant name. It is usually the largest text at the top, on the header, logo, or watermark of the menu. You MUST return this.
        STEP 2: Extract every food/drink item, its weight/size/quantity if shown, and its price exactly as written. Do NOT convert or change the currency.

        Return ONLY a raw JSON object in EXACTLY this format, nothing else:
        {
          "restaurant": "The Restaurant Name Here",
          "items": [{"item": "Item Name", "weight": "250g", "price": 150000}]
        }

        Rules:
        - "restaurant" must be a string — never null or empty. If truly not visible, make your best guess from any branding visible.
        - "weight" is the portion size, weight, or volume shown on the menu (e.g. "250g", "500ml", "1kg", "Large", "per piece"). Use null if not mentioned.
        - "price" must be a plain number with no currency symbols, spaces, or commas.
        - Do NOT wrap the response in markdown or code blocks.
        """

        client = Groq()
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )

        cleaned_text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()

        result = json.loads(cleaned_text)

        # Handle case where AI returns a flat array instead of an object
        if isinstance(result, list):
            items = result
            restaurant = filename
        else:
            restaurant = result.get("restaurant") or None
            items = result.get("items", [])

        # Clean up restaurant name
        if not restaurant or str(restaurant).lower() in ("null", "unknown", "n/a", ""):
            restaurant = filename

        df = pd.DataFrame(items)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        return df, restaurant
    except Exception as e:
        st.error(f"Error analyzing image '{filename}': {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, filename

# -----------------------------------------
# SIDEBAR: UPLOAD & ANALYZE
# -----------------------------------------
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

with st.sidebar:
    st.header("📸 Upload & Analyze Menus")
    if st.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

    tab_files, tab_folder = st.tabs(["Individual Files", "Folder Path"])

    images_to_process = []  # list of (name, file_ref) — file_ref is UploadedFile or Path

    with tab_files:
        uploaded_files = st.file_uploader("Choose images", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)
        if uploaded_files:
            images_to_process = [(f.name, f) for f in uploaded_files]
            st.success(f"{len(images_to_process)} file(s) ready.")

    with tab_folder:
        folder_path = st.text_input("Paste folder path", placeholder="/Users/you/menus/")
        if folder_path:
            folder = Path(folder_path)
            if folder.is_dir():
                found = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
                if found:
                    images_to_process = [(p.name, p) for p in found]
                    st.success(f"{len(images_to_process)} image(s) found in folder.")
                else:
                    st.warning("No images found in that folder.")
            else:
                st.error("Path not found or is not a folder.")

    if images_to_process:
        if st.button("🔍 Analyze Menus"):
            with st.spinner(f"Analyzing {len(images_to_process)} image(s) in parallel... please wait."):
                all_extracted_data = []
                progress = st.progress(0)
                completed = 0

                # Read images in main thread (UploadedFile is not thread-safe)
                loaded = [(name, Image.open(file_ref).copy()) for name, file_ref in images_to_process]

                def process_one(name, img):
                    return name, img, analyze_menu_image(img, filename=name)

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(process_one, name, img): name
                               for name, img in loaded}

                    for future in as_completed(futures):
                        name, img, (df_part, restaurant) = future.result()
                        st.image(img, caption=f"Done: {restaurant}", use_container_width=True)
                        if df_part is not None:
                            df_part['Source Menu'] = restaurant
                            all_extracted_data.append(df_part)
                        completed += 1
                        progress.progress(completed / len(images_to_process))

                if all_extracted_data:
                    st.session_state['extracted_data'] = pd.concat(all_extracted_data, ignore_index=True)
                    st.success("Analysis Complete!")

# -----------------------------------------
# SIMILARITY MATCHING
# -----------------------------------------
def _similarity(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def group_similar_items(df, threshold=0.6):
    """Clusters similar item names across menus. Returns groups with items from 2+ menus."""
    rows = df[['item', 'price', 'Source Menu']].copy().reset_index(drop=True)
    group_indices = []   # list of lists of row indices
    group_labels = []    # canonical (first-seen) name per group

    for idx, row in rows.iterrows():
        norm = row['item'].lower().strip()
        matched = False
        for g_idx, label in enumerate(group_labels):
            if _similarity(norm, label) >= threshold:
                group_indices[g_idx].append(idx)
                matched = True
                break
        if not matched:
            group_indices.append([idx])
            group_labels.append(norm)

    result = []
    for indices, label in zip(group_indices, group_labels):
        group_df = rows.loc[indices]
        if group_df['Source Menu'].nunique() >= 2:
            result.append((label.title(), group_df.reset_index(drop=True)))
    return result


# -----------------------------------------
# ANALYTICS DATA LOADING
# -----------------------------------------
TRANSACTIONS_PATH = Path(__file__).parent / "data" / "aziz_transactions.csv"

@st.cache_data
def load_transactions():
    df = pd.read_csv(TRANSACTIONS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'total_amount_usd': 'total_amount'})
    return df

ITEM_CATEGORIES = {
    'Shawarma Chicken Wrap': 'Wraps & Sandwiches',
    'Shawarma Meat Wrap':    'Wraps & Sandwiches',
    'Falafel Sandwich':      'Wraps & Sandwiches',
    'Kafta Sandwich':        'Wraps & Sandwiches',
    'Zaatar Manakeesh':      'Manakeesh',
    'Cheese Manakeesh':      'Manakeesh',
    'Shawarma Chicken':      'Grills',
    'Shawarma Meat':         'Grills',
    'Kafta Grilled':         'Grills',
    'Mixed Grill Plate':     'Grills',
    'Chicken Tawook':        'Grills',
    'Hummus':                'Mezze & Salads',
    'Mutabbal':              'Mezze & Salads',
    'Tabbouleh':             'Mezze & Salads',
    'Fattoush':              'Mezze & Salads',
    'Falafel Plate':         'Mezze & Salads',
    'Warak Arish (Stuffed Vine Leaves)': 'Mezze & Salads',
    'Foul Mdammas':          'Mezze & Salads',
    'Labneh':                'Mezze & Salads',
    'Shanklish Salad':       'Mezze & Salads',
    'Kibbeh Nayeh':          'Mezze & Salads',
    'Baklava (3 pcs)':       'Desserts',
    'Knafeh':                'Desserts',
    'Maamoul (2 pcs)':       'Desserts',
    'Rice Pudding':          'Desserts',
    'Ashta Cream':           'Desserts',
    'Sambousek (4 pcs)':     'Desserts',
    'Arabic Coffee':         'Beverages',
    'Lebanese Tea':          'Beverages',
    'Lemon Mint Juice':      'Beverages',
    'Jallab':                'Beverages',
    'Tamarind Juice':        'Beverages',
    'Ayran':                 'Beverages',
    'Soft Drink Can':        'Beverages',
    'Still Water 500ml':     'Beverages',
    'Olives Mix (250g)':     'Packaged Goods',
    'Pickled Turnip Jar':    'Packaged Goods',
    'Homemade Labneh Jar':   'Packaged Goods',
    'Tahini Paste (500g)':   'Packaged Goods',
    'Zaatar Spice Mix (100g)': 'Packaged Goods',
}

BRANCH_COLORS = {
    'Kantari': '#e07b39',
    'Zalka':   '#2a9d8f',
}

@st.cache_data
def parse_item_rows(df):
    """Expand items_purchased into one row per item with quantity and branch."""
    records = []
    for _, row in df.iterrows():
        for entry in row['items_purchased'].split(' | '):
            m = re.match(r'^(.+?)\s*\(x(\d+)\)$', entry.strip())
            if m:
                item = m.group(1).strip()
                records.append({
                    'date':          row['date'],
                    'branch':        row['location_name'],
                    'item':          item,
                    'category':      ITEM_CATEGORIES.get(item, 'Other'),
                    'quantity':      int(m.group(2)),
                    'total_amount':  row['total_amount'],
                })
    return pd.DataFrame(records)


# -----------------------------------------
# MAIN DASHBOARD AREA
# -----------------------------------------
logo_path = Path(__file__).parent / "data" / "aziz-artguru.png"
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(str(logo_path), width=200)
with col_title:
    st.title("Aziz Delicatesse: Competitor Price Intelligence")
    st.markdown("Automated menu extraction and competitor tracking.")
st.markdown("---")

tab_intel, tab_analytics = st.tabs(["🔍 Price Intelligence", "📊 Analytics"])

# ── TAB 1: PRICE INTELLIGENCE ──────────────────────────────────────────────
with tab_intel:
    if 'extracted_data' in st.session_state and not st.session_state['extracted_data'].empty:
        current_df = st.session_state['extracted_data']

        with st.expander("📋 All Extracted Raw Data", expanded=False):
            st.dataframe(current_df, use_container_width=True)

        st.markdown("---")

        threshold = st.slider("Similarity threshold for matching items", 0.3, 1.0, 0.6, 0.05,
                              help="Lower = more lenient matching. Raise if unrelated items are being grouped.")

        groups = group_similar_items(current_df, threshold)

        if groups:
            st.subheader(f"🔍 {len(groups)} Matched Item Group(s) Across Menus")

            # Assign a distinct colour to each restaurant
            RESTAURANT_COLORS = ['#d4edda', '#dce8f5', '#fde8d8', '#e8d8fd', '#fdf5d8', '#d8f5f0']
            all_restaurants = sorted({r['Source Menu'] for _, g in groups for _, r in g.iterrows()})
            col_colors = {rest: RESTAURANT_COLORS[i % len(RESTAURANT_COLORS)]
                          for i, rest in enumerate(all_restaurants)}

            # Build pivot table
            table_rows = []
            for label, group_df in groups:
                row = {'Item': label}
                for _, r in group_df.iterrows():
                    price_str = f"LBP {r['price']:,.0f}"
                    if 'weight' in r and pd.notna(r['weight']) and r['weight']:
                        price_str += f"  ({r['weight']})"
                    row[r['Source Menu']] = price_str
                table_rows.append(row)

            pivot_df = pd.DataFrame(table_rows).set_index('Item').fillna('—')

            # Apply column colours via Styler
            def color_columns(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                for col in df.columns:
                    if col in col_colors:
                        styles[col] = f'background-color: {col_colors[col]};'
                return styles

            styled = pivot_df.style.apply(color_columns, axis=None)
            st.dataframe(styled, use_container_width=True)

            # Colour legend
            legend_cols = st.columns(len(all_restaurants))
            for i, rest in enumerate(all_restaurants):
                legend_cols[i].markdown(
                    f'<div style="background:{col_colors[rest]};padding:6px 10px;'
                    f'border-radius:6px;text-align:center;font-weight:600;">{rest}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.subheader("Price Comparison Chart")

            # Build long-form dataframe for chart
            chart_rows = []
            for label, group_df in groups:
                for _, r in group_df.iterrows():
                    chart_rows.append({
                        'Item': label,
                        'Restaurant': r['Source Menu'],
                        'Price': r['price'],
                    })
            chart_df = pd.DataFrame(chart_rows)

            # Use solid chart colours (distinct from the pastel table colours)
            CHART_COLORS = ['#2a9d8f', '#e07b39', '#e63946', '#457b9d', '#f4a261', '#6a4c93']
            chart_color_map = {rest: CHART_COLORS[i % len(CHART_COLORS)]
                               for i, rest in enumerate(all_restaurants)}

            fig = px.bar(
                chart_df, x='Item', y='Price', color='Restaurant',
                barmode='group', text='Price',
                title="Price Comparison Across Restaurants",
                color_discrete_map=chart_color_map,
                labels={'Price': 'Price (LBP)', 'Item': 'Menu Item'}
            )
            fig.update_traces(texttemplate='LBP %{text:,.0f}', textposition='outside')
            fig.update_layout(margin=dict(t=50, b=20), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No matching items found across menus. Try lowering the similarity threshold above.")
    else:
        st.info("Upload menu images in the sidebar and click 'Analyze' to get started.")


# ── TAB 2: ANALYTICS ───────────────────────────────────────────────────────
with tab_analytics:
    col_a_logo, col_a_title = st.columns([1, 7])
    with col_a_logo:
        st.image(str(logo_path), width=100)
    with col_a_title:
        st.subheader("Aziz Delicatesse — Sales Analytics (Last 7 Days)")

    txn_df = load_transactions()
    items_df = parse_item_rows(txn_df)

    max_date = txn_df['date'].max()
    cutoff   = max_date - pd.Timedelta(days=6)
    all_dates = pd.date_range(cutoff, max_date, freq='D')

    txn_7   = txn_df[txn_df['date'] >= cutoff]
    items_7 = items_df[items_df['date'] >= cutoff]
    branches = sorted(txn_df['location_name'].unique())

    # ── Dashboard 1: Daily Sales Trend by Branch ────────────────────────
    with st.container(border=True):
        st.markdown("#### Daily Sales Trend — by Branch")

        daily_branch = (
            txn_7.groupby(['date', 'location_name'])['total_amount']
            .sum()
            .reset_index()
            .rename(columns={'location_name': 'Branch', 'total_amount': 'Revenue (USD)'})
        )
        daily_branch['date'] = daily_branch['date'].dt.strftime('%b %d')

        # Summary metrics per branch
        cols = st.columns(len(branches) + 1)
        total_rev = txn_7['total_amount'].sum()
        cols[0].metric("Total Revenue (7d)", f"USD {total_rev:,.0f}")
        for i, branch in enumerate(branches):
            branch_rev = txn_7[txn_7['location_name'] == branch]['total_amount'].sum()
            short = branch
            cols[i + 1].metric(f"{short}", f"USD {branch_rev:,.0f}")

        fig_sales = px.line(
            daily_branch, x='date', y='Revenue (USD)', color='Branch',
            markers=True, title="Daily Revenue — Last 7 Days",
            color_discrete_map=BRANCH_COLORS,
            labels={'date': 'Date'}
        )
        fig_sales.update_traces(marker=dict(size=8))
        fig_sales.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_sales, use_container_width=True)

    st.markdown("---")

    # ── Dashboard 2: Category Sales by Branch ───────────────────────────
    with st.container(border=True):
        st.markdown("#### Category Sales — Last 7 Days")

        all_cats = sorted(items_7['category'].unique())
        selected_cat = st.selectbox("Select a category", all_cats)

        cat_daily = (
            items_7[items_7['category'] == selected_cat]
            .groupby(['date', 'branch'])['quantity']
            .sum()
            .reset_index()
        )
        # Fill missing date × branch combos with 0
        idx = pd.MultiIndex.from_product([all_dates, branches], names=['date', 'branch'])
        cat_daily = (
            cat_daily.set_index(['date', 'branch'])
            .reindex(idx, fill_value=0)
            .reset_index()
        )
        cat_daily['date'] = cat_daily['date'].dt.strftime('%b %d')

        total_cat = cat_daily['quantity'].sum()
        st.metric(f"Total '{selected_cat}' units sold (7d)", f"{int(total_cat):,} units")

        fig_cat = px.bar(
            cat_daily, x='date', y='quantity', color='branch',
            barmode='group', text='quantity',
            title=f"Daily Units Sold — {selected_cat}",
            color_discrete_map=BRANCH_COLORS,
            labels={'date': 'Date', 'quantity': 'Units Sold', 'branch': 'Branch'}
        )
        fig_cat.update_traces(textposition='outside')
        fig_cat.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    # ── Dashboard 3: Item Deep-Dive by Branch ───────────────────────────
    with st.container(border=True):
        st.markdown("#### Menu Item Deep-Dive — Sales by Branch")

        all_items = sorted(items_7['item'].unique())
        selected_item = st.selectbox("Search and select a menu item", all_items)

        item_daily = (
            items_7[items_7['item'] == selected_item]
            .groupby(['date', 'branch'])['quantity']
            .sum()
            .reset_index()
        )
        idx2 = pd.MultiIndex.from_product([all_dates, branches], names=['date', 'branch'])
        item_daily = (
            item_daily.set_index(['date', 'branch'])
            .reindex(idx2, fill_value=0)
            .reset_index()
        )
        item_daily['date'] = item_daily['date'].dt.strftime('%b %d')

        # Metrics per branch
        cols2 = st.columns(len(branches))
        for i, branch in enumerate(branches):
            sold = item_daily[item_daily['branch'] == branch]['quantity'].sum()
            short = branch
            cols2[i].metric(f"{short}", f"{int(sold):,} units")

        fig_item = px.bar(
            item_daily, x='date', y='quantity', color='branch',
            barmode='group', text='quantity',
            title=f"Daily Units Sold — {selected_item} — by Branch",
            color_discrete_map=BRANCH_COLORS,
            labels={'date': 'Date', 'quantity': 'Units Sold', 'branch': 'Branch'}
        )
        fig_item.update_traces(textposition='outside')
        fig_item.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_item, use_container_width=True)