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
from supabase import create_client

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
    /* ── FORCE LIGHT THEME ALWAYS ── */
    .stApp { background-color: #FCFCFC !important; color: #1c1c1c !important; }
    h1, h2, h3, h4, p, span, label, div { color: #1c1c1c !important; font-family: 'Georgia', serif; }
    .stDataFrame { border: 1px solid #eaeaea; border-radius: 5px; }
    .css-1d391kg { background-color: #f4f4f4; }


    /* Force metric values and labels to be visible */
    [data-testid="stMetricValue"] { color: #1c1c1c !important; font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { color: #555555 !important; }
    [data-testid="stMetricDelta"] { color: #555555 !important; }

    /* Tabs text */
    .stTabs [data-baseweb="tab"] { color: #1c1c1c !important; }
    .stTabs [aria-selected="true"] { color: #e07b39 !important; }

    /* Sidebar — force light background and dark text */
    [data-testid="stSidebar"] { background-color: #f4f4f4 !important; }
    [data-testid="stSidebar"] * { color: #1c1c1c !important; }
    [data-testid="stSidebar"] .stButton > button { background-color: #ffffff !important; border: 1px solid #cccccc !important; }

    /* File uploader — force light on every internal element */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileDropzone"],
    [data-testid="stFileDropzone"] > div,
    .st-emotion-cache-u8yk7a,
    [data-testid="stFileUploader"] label { background-color: #ffffff !important; }
    [data-testid="stFileUploader"] * { color: #1c1c1c !important; }
    [data-testid="stFileDropzone"] { border: 2px dashed #aaaaaa !important; border-radius: 8px !important; }

    /* Browse files button inside uploader */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileDropzone"] button {
        background-color: #f0f0f0 !important;
        color: #1c1c1c !important;
        border: 1px solid #aaaaaa !important;
    }

    /* Selectbox — force light background and dark text */
    [data-baseweb="select"] * { background-color: #ffffff !important; color: #1c1c1c !important; }
    [data-baseweb="popover"] * { background-color: #ffffff !important; color: #1c1c1c !important; }
    [data-baseweb="menu"] { background-color: #ffffff !important; }
    [role="option"] { background-color: #ffffff !important; color: #1c1c1c !important; }
    [role="option"]:hover { background-color: #f0f0f0 !important; }

    /* Markdown text */
    .stMarkdown, .stMarkdown p { color: #1c1c1c !important; }

    /* Demo badge — must override the global span rule */
    span.demo-badge { color: #e00000 !important; }

    /* ── MOBILE RESPONSIVE ── */
    @media (max-width: 768px) {

        /* Remove excessive padding so content fills the screen */
        .main .block-container {
            padding: 0.75rem 0.75rem 1rem 0.75rem !important;
            max-width: 100% !important;
        }

        /* Stack all columns vertically */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        /* Bigger touch targets for buttons */
        .stButton > button {
            height: 3rem !important;
            font-size: 1rem !important;
            width: 100% !important;
        }

        /* Larger text inputs on mobile */
        input[type="text"], input[type="password"] {
            font-size: 1rem !important;
            height: 2.8rem !important;
        }

        /* Shrink logo on mobile so it doesn't dominate */
        img { max-width: 120px !important; }

        /* Give charts a minimum height for readability */
        .js-plotly-plot { min-height: 320px; }

        /* Reduce heading sizes */
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
    }
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
        You are an expert data extractor. Look at this restaurant/patisserie menu image carefully.
        The prices on this menu are in US Dollars (USD). Extract them as decimal USD values.

        STEP 1: Find the restaurant name. It is usually the largest text at the top, on the header, logo, or watermark of the menu. You MUST return this.
        STEP 2: Extract every food/drink item, its weight/size if shown, and its USD price.

        Return ONLY a raw JSON object in EXACTLY this format, nothing else:
        {
          "restaurant": "The Restaurant Name Here",
          "items": [{"item": "Item Name", "weight": "250g", "price": 12.50}]
        }

        Rules:
        - "restaurant" must be a string — never null or empty. If truly not visible, make your best guess from any branding visible.
        - "weight" must ONLY be a gram/kilogram/millilitre/litre measurement (e.g. "250g", "500ml", "1kg", "1.5kg"). Use null if the item is sold per piece, per portion, or no weight is shown.
        - "price" must be the USD dollar amount as a plain decimal number (e.g. 12.50). No currency symbols, no commas.
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
# SUPABASE PERSISTENCE
# -----------------------------------------
@st.cache_resource
def get_supabase():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        return None

def save_to_sheet(df):
    client = get_supabase()
    if client is None:
        st.warning("Supabase not connected — data not saved.")
        return
    try:
        date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        rows = []
        for _, row in df.iterrows():
            w = row.get('weight', '')
            rows.append({
                "date": date_str,
                "source_menu": str(row.get('Source Menu', '')),
                "item": str(row.get('item', '')),
                "weight": str(w) if pd.notna(w) and w != '' else None,
                "price": float(row['price'])
            })
        if rows:
            client.table("price_intelligence").insert(rows).execute()
    except Exception as e:
        st.error(f"Failed to save to database: {e}")

@st.cache_data(ttl=30)
def load_from_sheet():
    client = get_supabase()
    if client is None:
        return pd.DataFrame(columns=['date', 'Source Menu', 'item', 'weight', 'price'])
    try:
        response = client.table("price_intelligence").select("*").order("date", desc=True).execute()
        if not response.data:
            return pd.DataFrame(columns=['date', 'Source Menu', 'item', 'weight', 'price'])
        df = pd.DataFrame(response.data)
        df = df.rename(columns={'source_menu': 'Source Menu'})
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        return df[['date', 'Source Menu', 'item', 'weight', 'price']]
    except Exception as e:
        st.error(f"Failed to load from database: {e}")
        return pd.DataFrame(columns=['date', 'Source Menu', 'item', 'weight', 'price'])

# -----------------------------------------
# SIDEBAR: UPLOAD & ANALYZE
# -----------------------------------------
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

with st.sidebar:
    st.header("📸 Upload & Analyze Menus")
    if st.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

    # images_to_process: list of (filename, file_ref, restaurant_label)
    images_to_process = []

    st.markdown("#### Aziz Delicatesse")
    aziz_files = st.file_uploader(
        "Upload Aziz menu images",
        type=['png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True,
        key="aziz_uploader"
    )
    if aziz_files:
        images_to_process += [(f.name, f, "Aziz") for f in aziz_files]
        st.success(f"{len(aziz_files)} Aziz image(s) ready.")

    st.markdown("---")
    st.markdown("#### Noura")
    noura_files = st.file_uploader(
        "Upload Noura menu images",
        type=['png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True,
        key="noura_uploader"
    )
    if noura_files:
        images_to_process += [(f.name, f, "Noura") for f in noura_files]
        st.success(f"{len(noura_files)} Noura image(s) ready.")

    if images_to_process:
        if st.button("🔍 Analyze Menus"):
            with st.spinner(f"Analyzing {len(images_to_process)} image(s) in parallel... please wait."):
                all_extracted_data = []
                progress = st.progress(0)
                completed = 0

                # Read images in main thread (UploadedFile is not thread-safe)
                loaded = [(name, Image.open(file_ref).copy(), label) for name, file_ref, label in images_to_process]

                def process_one(name, img, label):
                    return name, img, label, analyze_menu_image(img, filename=name)

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(process_one, name, img, label): name
                               for name, img, label in loaded}

                    for future in as_completed(futures):
                        name, img, label, (df_part, _) = future.result()
                        st.image(img, caption=f"Done: {label}", use_container_width=True)
                        if df_part is not None:
                            df_part['Source Menu'] = label
                            all_extracted_data.append(df_part)
                        completed += 1
                        progress.progress(completed / len(images_to_process))

                if all_extracted_data:
                    st.session_state['extracted_data'] = pd.concat(all_extracted_data, ignore_index=True)
                    save_to_sheet(st.session_state['extracted_data'])
                    load_from_sheet.clear()
                    st.success("Analysis Complete! Data saved.")

# -----------------------------------------
# SIMILARITY MATCHING
# -----------------------------------------
def _similarity(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def parse_weight_grams(weight_str):
    """Parse weight strings like '150g', '1kg', '500ml', '1.5l' → float grams/ml.
    Returns None for anything that isn't a mass/volume unit (e.g. pieces, portions)."""
    if not weight_str or pd.isna(weight_str):
        return None
    s = str(weight_str).strip().lower()
    m = re.match(r'^([\d.]+)\s*(kg|g|ml|l)$', s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit in ('kg', 'l'):
        return val * 1000
    return val  # g or ml


def normalize_group_prices(group_df):
    """Normalize each item's price to the minimum weight found in the group.
    Items without a parseable weight are left as-is (display_price == price).
    Returns group_df with added columns: display_price, compare_weight."""
    cols = list(group_df.columns)
    weight_col = group_df['weight'] if 'weight' in cols else pd.Series([None] * len(group_df))
    weights_g = weight_col.apply(parse_weight_grams)
    valid = weights_g.dropna()

    group_df = group_df.copy()
    group_df['_wg'] = weights_g

    if valid.empty:
        group_df['display_price'] = group_df['price']
        group_df['compare_weight'] = None
    else:
        min_w = valid.min()
        group_df['display_price'] = group_df.apply(
            lambda r: r['price'] * (min_w / r['_wg']) if pd.notna(r['_wg']) else r['price'],
            axis=1
        )
        group_df['compare_weight'] = group_df['_wg'].apply(
            lambda w: min_w if pd.notna(w) else None
        )
    return group_df.drop(columns=['_wg'])


def group_similar_items(df, threshold=0.6):
    """Clusters similar item names across menus. Returns groups with items from 2+ menus."""
    keep_cols = ['item', 'price', 'Source Menu']
    if 'weight' in df.columns:
        keep_cols.append('weight')
    rows = df[keep_cols].copy().reset_index(drop=True)
    group_indices = []
    group_labels = []

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
    st.markdown(
        "# Aziz Delicatesse: Competitor Price Intelligence "
        "<span class='demo-badge' style='font-size:1rem;font-weight:400;'>(demo)</span>",
        unsafe_allow_html=True
    )
    st.markdown("Automated menu extraction and competitor tracking.")
st.markdown("---")

tab_intel, tab_analytics = st.tabs(["🔍 Price Intelligence", "📊 Analytics"])

# ── TAB 1: PRICE INTELLIGENCE ──────────────────────────────────────────────
with tab_intel:
    sheet_df = load_from_sheet()

    if not sheet_df.empty:
        available_dates = sorted(sheet_df['date'].unique(), reverse=True)
        selected_date = st.selectbox("📅 Select date", available_dates, index=0)
        current_df = sheet_df[sheet_df['date'] == selected_date].drop(columns=['date']).reset_index(drop=True)
    elif 'extracted_data' in st.session_state and not st.session_state['extracted_data'].empty:
        # Fallback to session state if sheets not configured
        current_df = st.session_state['extracted_data']
    else:
        current_df = pd.DataFrame()

    if not current_df.empty:

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

            # Build pivot table with weight-normalized USD prices
            table_rows = []
            for label, group_df in groups:
                normalized = normalize_group_prices(group_df)
                row = {'Item': label}
                for _, r in normalized.iterrows():
                    price_str = f"${r['display_price']:.2f}"
                    cw = r.get('compare_weight')
                    if pd.notna(cw) and cw:
                        w_label = f"{int(cw)}g" if cw == int(cw) else f"{cw}g"
                        price_str += f" / {w_label}"
                    row[r['Source Menu']] = price_str
                table_rows.append(row)

            pivot_df = pd.DataFrame(table_rows).set_index('Item').fillna('—')

            # Render as HTML table so CSS can fully control light/dark styling
            header_cells = "<th style='background:#2a2a2a;color:#ffffff;padding:8px 12px;text-align:left;'>Item</th>"
            for col in pivot_df.columns:
                bg = col_colors.get(col, '#ffffff')
                header_cells += f"<th style='background:{bg};color:#1c1c1c;padding:8px 12px;text-align:left;border-left:1px solid #ddd;'>{col}</th>"

            body_rows = ""
            for i, (item, row) in enumerate(pivot_df.iterrows()):
                row_bg = "#ffffff" if i % 2 == 0 else "#f9f9f9"
                body_rows += f"<tr><td style='background:{row_bg};color:#1c1c1c;padding:8px 12px;font-weight:600;border-top:1px solid #eee;'>{item}</td>"
                for col in pivot_df.columns:
                    cell_bg = col_colors.get(col, row_bg)
                    body_rows += f"<td style='background:{cell_bg};color:#1c1c1c;padding:8px 12px;border-left:1px solid #ddd;border-top:1px solid #eee;'>{row[col]}</td>"
                body_rows += "</tr>"

            html_table = f"""
            <div style='overflow-x:auto;border-radius:8px;border:1px solid #ddd;'>
            <table style='width:100%;border-collapse:collapse;font-family:Georgia,serif;font-size:0.9rem;'>
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{body_rows}</tbody>
            </table></div>"""
            st.markdown(html_table, unsafe_allow_html=True)



            st.markdown("---")
            st.subheader("Price Comparison Chart")

            # Build long-form dataframe for chart using weight-normalized USD prices
            chart_rows = []
            for label, group_df in groups:
                normalized = normalize_group_prices(group_df)
                for _, r in normalized.iterrows():
                    cw = r.get('compare_weight')
                    weight_note = ""
                    if pd.notna(cw) and cw:
                        w_label = f"{int(cw)}g" if cw == int(cw) else f"{cw}g"
                        weight_note = f" / {w_label}"
                    chart_rows.append({
                        'Item': label + weight_note,
                        'Restaurant': r['Source Menu'],
                        'Price (USD)': round(r['display_price'], 2),
                    })
            chart_df = pd.DataFrame(chart_rows)

            # Keep only the 3 items with the largest price difference across restaurants
            price_spread = (
                chart_df.groupby('Item')['Price (USD)']
                .agg(lambda x: x.max() - x.min())
                .nlargest(3)
            )
            chart_df = chart_df[chart_df['Item'].isin(price_spread.index)]

            CHART_COLORS = ['#2a9d8f', '#e07b39', '#e63946', '#457b9d', '#f4a261', '#6a4c93']
            chart_color_map = {rest: CHART_COLORS[i % len(CHART_COLORS)]
                               for i, rest in enumerate(all_restaurants)}

            fig = px.bar(
                chart_df, x='Item', y='Price (USD)', color='Restaurant',
                barmode='group', text='Price (USD)',
                title="Price Comparison Across Restaurants (USD)",
                color_discrete_map=chart_color_map,
                labels={'Price (USD)': 'Price (USD)', 'Item': 'Menu Item'}
            )
            fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig.update_layout(margin=dict(t=50, b=20), xaxis_tickangle=-30, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        else:
            st.info("No matching items found across menus. Try lowering the similarity threshold above.")
    else:
        st.info("No data yet. Upload menu images in the sidebar and click 'Analyze' to get started.")


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
        fig_sales.update_layout(margin=dict(t=40, b=20), template="plotly_white")
        st.plotly_chart(fig_sales, use_container_width=True, config={"displayModeBar": False})

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
        fig_cat.update_layout(margin=dict(t=40, b=20), template="plotly_white")
        st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

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
        fig_item.update_layout(margin=dict(t=40, b=20), template="plotly_white")
        st.plotly_chart(fig_item, use_container_width=True, config={"displayModeBar": False})