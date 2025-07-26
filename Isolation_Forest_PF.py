import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import IsolationForest
import datetime
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Détection d’Anomalies sur Fonds",
    layout="wide",
    page_icon="🧠"
)

# ------------------------------------------------
# 🧠 EN-TÊTE DU PROJET
# ------------------------------------------------
st.markdown("""
# 🧠 Détection d’Anomalies sur Performances de Fonds  
##### *Une application IA/Python dédiée aux professionnels de la gestion d’actifs*
""")

st.markdown("---")

st.markdown("""
Bienvenue dans cette interface intelligente d’analyse de portefeuille.  
Ce projet s'appuie sur l’algorithme **Isolation Forest (conta/0,25)** pour identifier des comportements **anormaux**, **trop volatils** ou **soupçonnés de manipulation** dans les fonds d’investissement.

### 🎯 Objectifs :
- Identifier rapidement les **fonds suspects** (trop constants, trop volatils…)
- Repérer les **comportements déviants** ou incohérents
- Gagner du temps grâce à une **analyse automatisée**

ℹ️ Cette démonstration porte sur un portefeuille exemple de **15 fonds diversifiés**, répartis sur une période de **36 mois** (rendements mensuels).

---

### 🔬 Comprendre le Cœur du Modèle : `Isolation Forest`

""")

st.info("📌 *Isolation Forest est le moteur d’intelligence artificielle utilisé ici. Dépliez le bloc ci-dessous pour en savoir plus.*")

with st.expander("🧠💡 Pourquoi utiliser Isolation Forest ?"):
    st.markdown("""
- L’algorithme **Isolation Forest** fonctionne en isolant aléatoirement les points dans un espace multidimensionnel.  
- Les **observations isolées très rapidement** sont jugées **anormales**, car éloignées des clusters classiques.
- Il est **non-supervisé** (aucun label requis), **non-paramétrique** et **très efficace** sur des jeux de données financiers réels.
- Idéal pour détecter des fonds au comportement **bizarre**, **lissé**, **excessif**, ou **non naturel**.
""")

# ------------------------------------
# 🗂️ PRÉSENTATION DU PORTEFEUILLE EXEMPLE
# ------------------------------------
st.markdown("## 🗂️ Portefeuille Exemple – 15 Fonds Diversifiés")
st.markdown("""
Voici les 15 fonds inclus dans le portefeuille analysé.  
Ils couvrent différentes classes d’actifs (actions, obligations, immobilier) et zones géographiques.

🔍 Cette diversité permet une analyse robuste à travers plusieurs dimensions de risque et de performance.
""")

col1, col2, col3 = st.columns(3)

# 🎯 Données des 15 fonds
fonds = [
    ["VFINX", "Vanguard 500 Index"],
    ["VWELX", "Vanguard Wellington"],
    ["FPURX", "Fidelity Puritan"],
    ["TRBCX", "T. Rowe Price Blue Chip"],
    ["DODGX", "Dodge & Cox Stock"],
    ["FCNTX", "Fidelity Contrafund"],
    ["VIMAX", "Vanguard Mid-Cap"],
    ["VSMAX", "Vanguard Small-Cap"],
    ["VGTSX", "Vanguard International Stocks"],
    ["VTABX", "Vanguard Int'l Bonds"],
    ["VBTLX", "Vanguard US Bonds"],
    ["DODFX", "Dodge & Cox Int'l"],
    ["PRHSX", "T. Rowe Health Sciences"],
    ["VGSIX", "Vanguard REIT"],
    ["QREARX", "AQR Real Estate Long/Short"]
]

# Création du DataFrame
df = pd.DataFrame(fonds, columns=["Ticker", "Nom du Fonds"])

# Fonction de style : zébrage, en-tête, alignements
def style_table(df):
    # Créer une DataFrame de styles vide
    zebra = pd.DataFrame('', index=df.index, columns=df.columns)
    zebra.loc[df.index % 2 == 0, :] = 'background-color: #e6f2ff'

    return df.style\
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('font-weight', 'bold'), ('text-align', 'left')]}
        ])\
        .set_properties(subset=["Ticker"], **{'text-align': 'center', 'width': '70px'})\
        .set_properties(subset=["Nom du Fonds"], **{'text-align': 'left'})\
        .apply(lambda _: zebra, axis=None)

# Affichage en trois colonnes Streamlit
col1, col2, col3 = st.columns(3)
for i, col in enumerate([col1, col2, col3]):
    start = i * 5
    end = start + 5
    sub_df = df.iloc[start:end].reset_index(drop=True)
    col.dataframe(style_table(sub_df), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 📥 Téléchargement et Traitement des Données")

# ------------------------------------
# 📥 TÉLÉCHARGEMENT DES DONNÉES
# ------------------------------------
tickers = [
    "VFINX", "VWELX", "FPURX", "TRBCX", "DODGX",
    "FCNTX", "VGSIX", "QREARX","VSMAX", "VGTSX", 
    "VBTLX", "DODFX", "PRHSX", "VIMAX", "VTABX"
]

@st.cache_data
def get_monthly_returns(tickers):
    # Période fixe : du 30 juin 2022 au 30 juin 2025
    start = datetime.datetime(2022, 6, 30)
    end = datetime.datetime(2025, 6, 30)

    # Téléchargement mensuel
    data = yf.download(
        tickers,
        start=start,
        end=end + datetime.timedelta(days=1),  # pour inclure le 30 juin
        interval='1mo',
        group_by='ticker',
        progress=False,
        auto_adjust=True
    )

    monthly_returns = pd.DataFrame()
    for ticker in tickers:
        if ticker in data:
            monthly_price = data[ticker]["Close"]
        else:
            monthly_price = data["Close"][ticker]
        monthly_returns[ticker] = monthly_price.pct_change()

    # Filtrage explicite des dates exactes (par précaution)
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns = monthly_returns[
        (monthly_returns.index >= pd.to_datetime("2022-06-30")) &
        (monthly_returns.index <= pd.to_datetime("2025-06-30"))
    ]

    return monthly_returns.dropna()

with st.spinner("Chargement des données sur 36 mois..."):
    returns = get_monthly_returns(tickers)
    st.success("Données chargées avec succès !")

# ------------------------------------
# 📈 Comparaison avec Benchmark (MSCI World) == donnees statiques !!!
# Définir la plage de dates
start_date = "2022-06-30"
end_date = "2025-06-30"

# Liste des 15 fonds
tickers = [
    "VFINX", "VWELX", "FPURX", "TRBCX", "DODGX",
    "FCNTX", "VGSIX", "QREARX","VSMAX", "VGTSX", 
    "VBTLX", "DODFX", "PRHSX", "VIMAX", "VTABX"
]

# Télécharger les données mensuelles
fund_data = yf.download(tickers, start=start_date, end=end_date, interval='1mo', group_by='ticker', auto_adjust=True, progress=False)

monthly_prices = pd.DataFrame()
for ticker in tickers:
    try:
        monthly_prices[ticker] = fund_data[ticker]["Close"]
    except:
        monthly_prices[ticker] = fund_data["Close"][ticker]

# Calcul portefeuille moyen (base 100)
portfolio_mean = monthly_prices.mean(axis=1)
portfolio_perf = (portfolio_mean / portfolio_mean.iloc[0]) * 100

# Benchmark MSCI World via URTH (base 100)
benchmark_data = yf.download("URTH", start=start_date, end=end_date, interval='1mo', auto_adjust=True, progress=False)
benchmark_price = benchmark_data["Close"]
benchmark_perf = (benchmark_price / benchmark_price.iloc[0]) * 100

# Construction DataFrame final
df_static = pd.DataFrame({
    "Date": portfolio_perf.index,
    "Portefeuille Moyen": portfolio_perf.values.flatten(),  # <- important si ndarray 2D
    "MSCI World": benchmark_perf.reindex(portfolio_perf.index).values.flatten()
})

df_static.reset_index(drop=True, inplace=True)

# 📈 Calcul Alpha & Beta de Jensen (régression sur les rendements)

# Calcul des rendements mensuels à partir des valeurs cumulées
df_static["Rendement Portefeuille"] = df_static["Portefeuille Moyen"].pct_change()
df_static["Rendement Benchmark"] = df_static["MSCI World"].pct_change()

excess_returns = df_static["Rendement Portefeuille"] - df_static["Rendement Benchmark"]
tracking_error = excess_returns.std()
mean_excess = excess_returns.mean()
info_ratio_port = mean_excess / tracking_error if tracking_error > 1e-6 else 0.0
info_ratio_port_str = f"{info_ratio_port:.2f}"

# Nettoyage des NaN
df_alpha = df_static.dropna(subset=["Rendement Portefeuille", "Rendement Benchmark"])

# Régression linéaire : Portefeuille = alpha + beta * Benchmark
from sklearn.linear_model import LinearRegression

X = df_alpha[["Rendement Benchmark"]].values
y = df_alpha["Rendement Portefeuille"].values

model = LinearRegression()
model.fit(X, y)

alpha = model.intercept_
beta = model.coef_[0]

# Stockage des résultats formatés
alpha_percent = f"{alpha * 100:.2f} %"
beta_value = f"{beta:.2f}"

# 🔁 Étape 2 : Tracer
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_static["Date"],
    y=df_static["Portefeuille Moyen"],
    mode='lines',
    name="Portefeuille Moyen",
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=df_static["Date"],
    y=df_static["MSCI World"],
    mode='lines',
    name="MSCI World",
    line=dict(color='red')
))

fig.update_layout(
    title="Comparaison Benchmark : Portefeuille vs MSCI World (Base 100)",
    xaxis_title="Date",
    yaxis_title="Valeur cumulée (base 100)",
    yaxis=dict(range=[80, df_static[["Portefeuille Moyen", "MSCI World"]].max().max() * 1.05]),
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig, use_container_width=True)

# 🔍 Calculs dynamiques des indicateurs de performance
rendements_portefeuille = df_static["Rendement Portefeuille"].dropna()
rendements_benchmark = df_static["Rendement Benchmark"].dropna()

# Calcul du Ratio d'information du portefeuille
excess_returns_port = rendements_portefeuille - rendements_benchmark
tracking_error_port = excess_returns_port.std()
mean_excess_port = excess_returns_port.mean()
info_ratio_port = mean_excess_port / tracking_error_port if tracking_error_port > 1e-6 else 0.0
info_ratio_port_str = f"{info_ratio_port:.2f}"

# Alignement des séries
min_len = min(len(rendements_portefeuille), len(rendements_benchmark))
rendements_portefeuille = rendements_portefeuille[-min_len:]
rendements_benchmark = rendements_benchmark[-min_len:]

# Statistiques dynamiques
moyenne_portefeuille = rendements_portefeuille.mean() * 12
moyenne_benchmark = rendements_benchmark.mean() * 12
vol_portefeuille = rendements_portefeuille.std() * np.sqrt(12)
vol_benchmark = rendements_benchmark.std() * np.sqrt(12)

sharpe_portefeuille = moyenne_portefeuille / vol_portefeuille
sharpe_benchmark = moyenne_benchmark / vol_benchmark

# Formatage pour affichage dans tableau
moyenne_portefeuille_str = f"{moyenne_portefeuille:.1%}"
moyenne_benchmark_str = f"{moyenne_benchmark:.1%}"
vol_portefeuille_str = f"{vol_portefeuille:.1%}"
vol_benchmark_str = f"{vol_benchmark:.1%}"
sharpe_portefeuille_str = f"{sharpe_portefeuille:.2f}"
sharpe_benchmark_str = f"{sharpe_benchmark:.2f}"

# Exemple de conversion mois -> jours
tt_recovery_portefeuille = f"{9 * 30} jours"
tt_recovery_benchmark = f"{6 * 30} jours"


# Les datas pour le tableau
data = {
    "Ratio": [
        "Rendement", "Volatilité", "Sharpe",
        "Information Ratio", "Alpha de Jensen*", "Beta",
        "Max Drawdown", "Time to Recovery (jours)"
    ],
    "Portefeuille": [
        moyenne_portefeuille_str, vol_portefeuille_str, sharpe_portefeuille_str,
        info_ratio_port_str, alpha_percent, beta_value,
        "-11 %", "43 jours"  
    ],
    "Benchmark": [
        moyenne_benchmark_str, vol_benchmark_str, sharpe_benchmark_str,
        "—", "—", "1.0",
        "-17 %", "92 jours"  
    ]
}

df = pd.DataFrame(data)

# Affichage dans Streamlit
st.markdown("📊 **Exemple comparatif simplifié :**")

st.table(df.set_index("Ratio"))
# 📝 Explication discrète de l'Alpha
st.markdown("*`* Alpha de Jensen → ajusté du risque (via régression linéaire sur les rendements mensuels).`*", unsafe_allow_html=True)


# 📈 4. Analyse Déviation vs Benchmark – CORRIGÉ
# ------------------------------------
st.markdown("### 📈 4. Analyse de la Déviation au Benchmark")

# Vérifie si benchmark URTH est bien dans returns (si pas, le reconstruire)
if "URTH" not in returns.columns:
    benchmark_data = yf.download("URTH", start=start_date, end=end_date, interval='1mo', auto_adjust=True, progress=False)
    benchmark = benchmark_data["Close"].pct_change().dropna()
else:
    benchmark = returns["URTH"].dropna()

# On s’aligne bien avec les rendements des fonds
benchmark.name = "Benchmark"
aligned = returns.copy()
aligned["Benchmark"] = benchmark
aligned = aligned.dropna()

# Initialisation
tracking_errors = []
information_ratios = []
sharpes = []
fonds = []

for col in aligned.columns:
    if col == "Benchmark":
        continue

    fund_returns = aligned[col]
    benchmark_returns = aligned["Benchmark"]

    # Calculs
    excess_returns = fund_returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    mean_excess = np.mean(excess_returns)
    info_ratio = mean_excess / tracking_error if tracking_error > 1e-6 else 0.0
    sharpe = fund_returns.mean() / fund_returns.std() * np.sqrt(12)

    fonds.append(col)
    tracking_errors.append(tracking_error)
    information_ratios.append(info_ratio)
    sharpes.append(sharpe)

# Création du DataFrame propre
df_ir = pd.DataFrame({
    "Fonds": fonds,
    "Tracking Error": tracking_errors,
    "Information Ratio": information_ratios,
    "Sharpe Ratio": sharpes
})

# Fusion avec les profils comportementaux (issus du clustering)
if "df_metrics" in locals() and "Profil" in df_metrics.columns:
    df_ir = df_ir.merge(df_metrics[["Fonds", "Profil"]], on="Fonds", how="left")
else:
    df_ir["Profil"] = "Inconnu"

# Affichage graphique
fig_ir = px.scatter(
    df_ir,
    x="Tracking Error",
    y="Information Ratio",
    color="Information Ratio",  # 👈 couleur évolutive selon la performance
    size=df_ir["Sharpe Ratio"].apply(lambda x: x if x > 0 else 0),
    text="Fonds",
    title="🎯 Déviation vs Benchmark : Tracking Error vs Information Ratio",
    labels={
        "Tracking Error": "Tracking Error (écart-type vs Benchmark)",
        "Information Ratio": "Information Ratio"
    },
    hover_data=["Sharpe Ratio"],
    color_continuous_scale="Viridis"  # 👈 palette évolutive
)

fig_ir.update_traces(textposition="top center")
st.plotly_chart(fig_ir, use_container_width=True)

# Commentaire automatique
st.markdown("#### 🧠 Interprétation rapide")
st.markdown("""
- Les **fonds situés en haut à gauche** (IR élevé, faible TE) sont **très efficaces** :  
  ils battent le benchmark **avec peu de dispersion**.
- Ceux en bas à droite sont à risque : ils prennent des écarts importants **sans surperformer**.
- La **taille du point** reflète la qualité du couple rendement/risque (Sharpe > 0 uniquement).
""")




# ------------------------------------
# 🚨 DÉTECTION D’ANOMALIES
# ------------------------------------
st.markdown("### 🚨 Détection d’Anomalies par Isolation Forest")

model = IsolationForest(contamination=0.25, random_state=42)
predictions = model.fit_predict(returns.T)
anomalies = pd.Series(predictions, index=returns.columns)
result_df = pd.DataFrame({
    "Fonds": anomalies.index,
    "Anomalie": ["🚨 Suspect" if x == -1 else "✅ OK" for x in anomalies]
})
st.dataframe(result_df.style.applymap(lambda x: "background-color: #ffcccc" if x == "🚨 Suspect" else "background-color: #ccffcc", subset=["Anomalie"]))


# 🚨 2.2. Zoom sur les Anomalies Détectées – Analyse Technique + Graphique + Explication
st.markdown("### 🚨 2.2. Zoom sur les Anomalies Détectées – Analyse Technique")

from scipy.stats import skew, kurtosis
import plotly.graph_objects as go

anomalies_detectees = result_df[result_df['Anomalie'] == "🚨 Suspect"]['Fonds'].tolist()

if anomalies_detectees:
    for i, fond in enumerate(anomalies_detectees[:4], start=1):
        rendements_fond = returns[fond].dropna()
        portefeuille_moyen = returns.mean(axis=1).dropna()
        portefeuille_moyen = portefeuille_moyen.loc[rendements_fond.index]

        # Analyse quantitative
        volatilite = rendements_fond.std()
        cum_perf = (1 + rendements_fond).cumprod()
        max_dd = (cum_perf.cummax() - cum_perf).max() / cum_perf.cummax().max()
        skewness = skew(rendements_fond)
        kurt = kurtosis(rendements_fond)

        st.markdown(f"#### ⚠️ Anomalie {i} – **{fond}**")
        st.markdown(f"""
📊 **Analyse Quantitative du Fonds `{fond}`**  
- **Volatilité** : {volatilite:.2%}  
- **Max Drawdown** : {max_dd:.2%}  
- **Asymétrie (Skewness)** : {skewness:.2f}  
- **Aplatissement (Kurtosis)** : {kurt:.2f}  
""")

        # Bloc analyse approfondie enrichi (version technique + pédagogique)
        st.markdown("🧠 **Analyse de l’anomalie :**")

        # Recalculs techniques utiles
        excess_returns = rendements_fond - portefeuille_moyen
        TE = np.std(excess_returns)
        info_ratio = excess_returns.mean() / TE if TE > 1e-6 else 0.0

        if volatilite > 0.07 and skewness < -0.5:
            st.markdown(f"""
- La **volatilité élevée ({volatilite:.2%})** indique une forte dispersion des rendements, souvent observée dans les stratégies spéculatives ou non diversifiées.
- L’**asymétrie négative prononcée ({skewness:.2f})** suggère une distribution des rendements biaisée vers les pertes extrêmes.
- Le **max drawdown ({max_dd:.2%})** confirme une baisse marquée non récupérée sur la période analysée.
- 👉 **Hypothèse** : Fonds orienté vers des actifs risqués ou avec effet de levier.
- ✅ **Recommandation** : Analyser la pondération sectorielle et les actifs sous-jacents.
            """)

        elif volatilite < 0.015 and kurt < 1:
            st.markdown(f"""
- Le niveau de volatilité ({volatilite:.2%}) est **anormalement faible** pour un fonds actions, ce qui soulève des doutes sur l'exposition réelle au marché, independemment du Real Estate.
- Une **kurtosis faible ({kurt:.2f})** implique une distribution très plate, sans queues épaisses, donc peu de variations extrêmes positives ou négatives.
- L’absence de volatilité couplée à un **drawdown significatif ({max_dd:.2%})** pourrait indiquer un effet de lissage dans la valorisation.
- 👉 **Hypothèse** : Comportement de type **window dressing** (ajuster temporairement le portefeuille à la fin d'une période) ou **valeur liquidative trop lissée** (problèmes de fréquence de pricing ou illiquidité).
- ✅ **Recommandation** : Vérifier la fréquence de pricing, la composition exacte du portefeuille, le processus de valorisation et les documents réglementaires (KID, prospectus).
            """)

        elif skewness > 0.8 and kurt < 2:
            st.markdown(f"""
- L’**asymétrie positive élevée ({skewness:.2f})** est rare et traduit une occurrence plus fréquente de rendements positifs que négatifs.
- La **kurtosis modérée ({kurt:.2f})** indique une faible dispersion extrême malgré une tendance à la hausse.
- Ce profil est typique de fonds concentrés sur quelques titres fortement performants, créant une illusion de stabilité avec biais haussier.
- 👉 **Hypothèse** : Stratégie biaisée vers quelques valeurs gagnantes ou effets ponctuels positifs.
- ✅ **Recommandation** : Surveiller la concentration en titres et la régularité des rendements.
            """)

        elif info_ratio < 0 and TE > 0.03:
            st.markdown(f"""
- L’asymétrie quasi nulle combinée à une faible kurtosis peut indiquer un **pilotage actif** pour contenir les extrêmes, typique des fonds qui utilisent des stratégies de couverture systématique (options collar...), ou encore d’une allocation sectorielle très dispersée pour **amortir les chocs**.
- La **Tracking Error élevée ({TE:.2%})** montre une forte divergence vis-à-vis du benchmark, ce qui renforce le **signal d’inefficacité**.
- Ce comportement peut être lié à **des arbitrages brutaux** dans la gestion.
- Ce profil est caractéristique de **stratégies actives mal calibrées**, où les paris du gérant ne créent pas de valeur.
- 👉 **Hypothèse** : Stratégie structurellement orientée vers la neutralité de marché, avec potentiellement des positions dérivées ou une gestion défensive visant à **stabiliser le mark-to-market**.
- ✅ **Recommandation** : Croiser les données avec des **statistiques intrajournalières** si disponibles, ou des **comparables** de même catégorie. Réévaluer sa place dans le portefeuille.
            """)

        else:
            st.markdown(f"""
- Ce fonds montre des écarts statistiques qui le distinguent des autres, **sans être extrêmes** sur un indicateur unique. L’anomalie provient d’un **cumul de petites déviations** sur plusieurs critères, plutôt que d’un écart important sur **une seule métrique**.
- Il peut s’agir d’un comportement de marché atypique, d’un changement de style de gestion ou d’une **dynamique propre au fonds**.
- 👉 **Hypothèse** : Comportement atypique ou changement de style de gestion.
- ✅ **Recommandation** : Compléter l’analyse avec des **indicateurs qualitatifs** (style de gestion, communication récente, rotation du portefeuille...).
            """)

        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rendements_fond.index, y=(1 + rendements_fond).cumprod(),
            name=f"{fond}", line=dict(color="red")))
        fig.add_trace(go.Scatter(
            x=portefeuille_moyen.index, y=(1 + portefeuille_moyen).cumprod(),
            name="Portefeuille Moyen", line=dict(color="blue", dash="dot")))
        fig.update_layout(
            title=f"📉 Performance Cumulée – {fond} vs Portefeuille Moyen",
            xaxis_title="Date", yaxis_title="Base 100",
            yaxis=dict(range=[0.7, max((1 + rendements_fond).cumprod().max(), (1 + portefeuille_moyen).cumprod().max()) * 1.05])
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.success("✅ Aucune anomalie détectée nécessitant une analyse technique.")

st.markdown("---")
st.markdown("Projet IA/Python appliqué à la finance – par Khemais Jelidi 💼")

