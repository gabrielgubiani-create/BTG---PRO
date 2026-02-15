import streamlit as st
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# ==========================================================
# 1. IDENTIDADE VISUAL & UI (ENTERPRISE HIGH-END)
# ==========================================================
st.set_page_config(page_title="POSTGIBBS PRO", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .header-pro {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        border-bottom: 4px solid #1f77b4;
    }
    .header-pro h1 { color: #ffffff !important; font-size: 1.6rem; margin: 0; letter-spacing: 1px; }
    .section-label {
        font-size: 0.85rem; font-weight: 700; color: #666;
        text-transform: uppercase; margin-bottom: 8px; margin-top: 15px;
    }
    .info-box {
        background-color: #f1f4f9; padding: 15px; border-radius: 8px;
        border-left: 5px solid #1f77b4; font-size: 0.9rem; margin-bottom: 15px;
    }
    .legend-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .legend-table td { padding: 5px; border-bottom: 1px solid #e0e0e0; font-size: 0.85rem; }
    .stNumberInput { margin-top: -15px; }
    .suggestion-box {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-pro"><h1>POSTGIBBS PRO</h1></div>', unsafe_allow_html=True)

# ==========================================================
# 2. ENTRADA DE DADOS E NOMENCLATURA
# ==========================================================
st.markdown('<div class="section-label">📁 Configuração e Arquivo</div>', unsafe_allow_html=True)
c_up, c_n, c_s = st.columns([2, 1, 1])
with c_up:
    uploaded_file = st.file_uploader("Upload 'postgibbs_samples'", label_visibility="collapsed")
with c_n:
    n_traits = st.number_input("Nº de Características", min_value=1, value=3)
with c_s:
    with st.expander("⚙️ Avançado"):
        skip_cols = st.number_input("Pular colunas", min_value=0, value=3)

st.markdown('<div class="section-label">🏷️ Nomenclatura das Características</div>', unsafe_allow_html=True)
trait_names_input = st.text_input("Insira os nomes separados por vírgula", placeholder="Ex: Stay, P120, P240", label_visibility="collapsed")

if trait_names_input:
    trait_names = [t.strip() for t in trait_names_input.split(",")][:n_traits]
    if len(trait_names) < n_traits:
        trait_names += [f"Trait{i+1}" for i in range(len(trait_names), n_traits)]
else:
    trait_names = [f"Trait{i+1}" for i in range(n_traits)]

# ==========================================================
# 3. MAPEAMENTO ESTRUTURAL
# ==========================================================
st.markdown('<div class="section-label">🏗️ Estrutura do Modelo</div>', unsafe_allow_html=True)
col_m, col_p = st.columns(2)
with col_m: show_mat = st.checkbox("Possui Efeito Materno?", value=False)
with col_p: show_pe = st.checkbox("Possui Efeito de Ambiente Permanente?", value=False)

has_mat, is_mat_full = [], False
has_pe, is_pe_full = [], False

if show_mat or show_pe:
    st.markdown("---")
    m1, m2 = st.columns(2)
    with m1:
        if show_mat:
            has_mat = st.multiselect("Selecione as Traits (Materno):", options=trait_names)
            with st.expander("Config. de Matriz (Mat)"): is_mat_full = st.checkbox("Matriz Completa", value=False)
    with m2:
        if show_pe:
            has_pe = st.multiselect("Selecione as Traits (PE):", options=trait_names)
            with st.expander("Config. de Matriz (PE)"): is_pe_full = st.checkbox("Matriz Completa", value=False, key="pe_full")

# Matemática de colunas
n_g = (n_traits * (n_traits + 1)) // 2
n_m = ((len(has_mat) * (len(has_mat) + 1)) // 2 if is_mat_full else len(has_mat)) if show_mat else 0
n_pe = ((len(has_pe) * (len(has_pe) + 1)) // 2 if is_pe_full else len(has_pe)) if show_pe else 0
n_r = (n_traits * (n_traits + 1)) // 2
total_expected = n_g + n_m + n_pe + n_r
st.caption(f"🏁 **Status:** Esperado {total_expected} colunas de dados.")

# ==========================================================
# 4. FUNÇÕES DE SUPORTE
# ==========================================================
def calculate_geweke(data):
    n = len(data)
    a, b = data[:int(0.1*n)], data[int(0.5*n):]
    if len(a) < 2 or len(b) < 2: return np.nan
    return (np.mean(a) - np.mean(b)) / (np.sqrt(np.var(a)/len(a) + np.var(b)/len(b)))

def suggest_burnin(data, step=500):
    best_burn = 0
    best_z = 999
    max_limit = int(len(data) * 0.5)
    for b in range(0, max_limit, step):
        z = abs(calculate_geweke(data[b:]))
        if not np.isnan(z) and z < best_z:
            best_z = z
            best_burn = b
    return best_burn

# ==========================================================
# 5. PROCESSAMENTO
# ==========================================================
if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
        df_data = df_raw.iloc[:, skip_cols:].reset_index(drop=True)
        max_samples = len(df_data)
        
        if df_data.shape[1] < total_expected:
            st.error(f"Incompatibilidade: Arquivo tem {df_data.shape[1]} colunas.")
            st.stop()
            
        current = 0
        df_G = df_data.iloc[:, current : current + n_g]; current += n_g
        if n_m > 0: df_M = df_data.iloc[:, current : current + n_m]; current += n_m
        if n_pe > 0: df_PE = df_data.iloc[:, current : current + n_pe]; current += n_pe
        df_R = df_data.iloc[:, current : current + n_r]

        df_params = pd.DataFrame()
        def get_inds(n): return [(i, j) for i in range(n) for j in range(i, n)]
        inds_f = get_inds(n_traits)

        for k in range(n_traits):
            name = trait_names[k]
            v_a = df_G.iloc[:, inds_f.index((k, k))]
            v_m = (df_M.iloc[:, get_inds(len(has_mat)).index((has_mat.index(name), has_mat.index(name)))] if is_mat_full else df_M.iloc[:, has_mat.index(name)]) if (show_mat and name in has_mat) else 0
            v_pe = (df_PE.iloc[:, get_inds(len(has_pe)).index((has_pe.index(name), has_pe.index(name)))] if is_pe_full else df_PE.iloc[:, has_pe.index(name)]) if (show_pe and name in has_pe) else 0
            v_e = df_R.iloc[:, inds_f.index((k, k))]
            v_p = v_a + v_m + v_pe + v_e
            
            df_params[f"h2_{name}"] = v_a / v_p
            if show_mat and name in has_mat: df_params[f"h2m_{name}"] = v_m / v_p
            if show_pe and name in has_pe:  df_params[f"h2c_{name}"] = v_pe / v_p
            df_params[f"σ²a_{name}"] = v_a
            if show_mat and name in has_mat: df_params[f"σ²m_{name}"] = v_m
            if show_pe and name in has_pe:  df_params[f"σ²c_{name}"] = v_pe
            df_params[f"σ²p_{name}"] = v_p
            df_params[f"σ²e_{name}"] = v_e

        st.markdown("---")
        t1, t2 = st.tabs(["📈 Convergência", "📄 Relatório Final"])

        with t1:
            # Lista todas as herdabilidades disponíveis (direta, materna e PE)
            h_ratio_cols = [c for c in df_params.columns if c.startswith(("h2_", "h2m_", "h2c_"))]
            if "burnin_dict" not in st.session_state: st.session_state["burnin_dict"] = {c: 0 for c in h_ratio_cols}
            
            c_sel, c_plot = st.columns([1.2, 3])
            with c_sel:
                sel = st.selectbox("Selecione o parâmetro para ajuste:", h_ratio_cols)
                suggested = suggest_burnin(df_params[sel].values)
                st.markdown(f"""<div class="suggestion-box">💡 <b>Sugestão:</b> {suggested} iterações</div>""", unsafe_allow_html=True)
                
                if st.button("Aplicar Sugestão"):
                    st.session_state["burnin_dict"][sel] = suggested
                    st.rerun()

                st.write("**Ajuste Manual de Burn-in**")
                slider_val = st.slider("Seletor Visual", 0, max_samples-100, 
                                     value=st.session_state["burnin_dict"].get(sel, 0), step=100, label_visibility="collapsed")
                manual_val = st.number_input("Definição Manual", min_value=0, max_value=max_samples-10, 
                                           value=slider_val, step=1)
                
                burn = manual_val
                st.session_state["burnin_dict"][sel] = burn
                z_gw = calculate_geweke(df_params[sel].iloc[burn:].values)
                if not np.isnan(z_gw): st.metric("Geweke Z-Score", f"{z_gw:.3f}")
            
            with c_plot:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={'width_ratios': [2, 1]})
                ax1.plot(df_params[sel], color='#e0e0e0', lw=0.5)
                ax1.plot(range(burn, max_samples), df_params[sel].iloc[burn:], color='#1f77b4', lw=1)
                ax1.axvline(burn, color='#d62728', ls='--')
                ax1.set_title(f"Cadeia: {sel}")
                az.plot_dist(df_params[sel].iloc[burn:].values, ax=ax2, color='#1f77b4', fill_kwargs={'alpha': 0.2})
                ax2.set_title("Densidade Posterior")
                st.pyplot(fig)

        with t2:
            summary = []
            prefixes = ["h2_", "h2m_", "h2c_", "σ²a", "σ²m", "σ²c", "σ²p", "σ²e"]
            
            for p in prefixes:
                cols_to_process = [c for c in df_params.columns if c.startswith(p)]
                for col in cols_to_process:
                    # Lógica de mapeamento de Burn-in:
                    # Tenta encontrar o burn-in da h2 correspondente à característica na coluna atual
                    trait_name = col.split("_", 1)[1]
                    
                    # Define qual burn-in usar baseado no tipo de variância
                    if p in ["h2_", "σ²a", "σ²p", "σ²e"]:
                        lookup_key = f"h2_{trait_name}"
                    elif p in ["h2m_", "σ²m"]:
                        lookup_key = f"h2m_{trait_name}"
                    elif p in ["h2c_", "σ²c"]:
                        lookup_key = f"h2c_{trait_name}"
                    else:
                        lookup_key = None
                    
                    b = st.session_state["burnin_dict"].get(lookup_key, 0)
                    
                    d = df_params[col].iloc[int(b):].values
                    s = az.summary(az.from_dict(posterior={col: d}), hdi_prob=0.95)
                    s["Geweke_Z"] = calculate_geweke(d)
                    s["burn_in"] = int(b)
                    s["n_samples"] = int(len(d))
                    summary.append(s)

            if summary:
                final_df = pd.concat(summary)
                st.dataframe(final_df[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "Geweke_Z", "ess_bulk", "burn_in", "n_samples"]].style.format({
                    "mean": "{:.4f}", "sd": "{:.4f}", "hdi_2.5%": "{:.4f}", "hdi_97.5%": "{:.4f}", "Geweke_Z": "{:.3f}", "burn_in": "{:d}", "n_samples": "{:d}"
                }), use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                    <b>📝 Legenda dos Parâmetros:</b>
                    <table class="legend-table">
                        <tr><td><b>h2</b>: Herdabilidade Direta ($h^2_a$)</td><td><b>σ²a</b>: Variância Genética Aditiva</td></tr>
                        <tr><td><b>h2m</b>: Herdabilidade Materna ($h^2_m$)</td><td><b>σ²m</b>: Variância Genética Materna</td></tr>
                        <tr><td><b>h2c</b>: Fração de Amb. Permanente ($c^2$)</td><td><b>σ²c</b>: Variância de Amb. Permanente</td></tr>
                        <tr><td><b>Geweke_Z</b>: Diagnóstico de Convergência (Ideal entre -1.96 e 1.96)</td><td><b>n_samples</b>: Nº de Amostras Válidas</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                st.download_button("📥 Baixar CSV Completo", final_df.to_csv().encode(), "BGT_PRO_Results.csv", use_container_width=True)
    except Exception as e: st.error(f"Erro: {e}")

else: st.info("Pronto para processar. Carregue o arquivo acima.")


