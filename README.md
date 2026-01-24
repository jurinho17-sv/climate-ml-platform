# Global CO2 Emissions Dashboard

An interactive dashboard to explore global CO2 emissions trends (1960-2023) and identify the fastest-growing emitters.

**Live Demo:** [global-co2-insight-by-ju-ho-kim.streamlit.app](https://global-co2-insight-by-ju-ho-kim.streamlit.app)

![Dashboard Screenshot](assets/dashboard_screenshot.png)

---

## Features

- **Volume Analysis:** Visualize top CO2 emitting countries with Area/Line charts
- **Growth Analysis:** Identify countries with the highest emission growth rates
- **Interactive Controls:** Filter by year range, select countries, toggle chart types
- **Robust Data Cleaning:** Multi-layer filtering to exclude non-country aggregates (e.g., "World", "OECD Members")

---

## Tech Stack

- **Language:** Python
- **Framework:** Streamlit
- **Visualization:** Plotly Express
- **Data Processing:** Pandas

---

## Project Structure

```
global-co2-insight/
├── data/                  # CO2 emissions dataset (1960-2023)
│   └── co2_emissions_kt_by_country_2023.csv
├── notebooks/             # EDA and data exploration
│   └── 01_data_eda.ipynb
├── src/                   # Streamlit application
│   └── app.py
├── assets/                # Screenshots for documentation
├── requirements.txt       # Python dependencies
└── README.md
```

---

## How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/jurinho17-sv/global-co2-insight.git
cd global-co2-insight
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run src/app.py
```

---

## Data Source

**World Bank / Our World in Data**  
- Dataset: [CO2 and Greenhouse Gas Emissions](https://data360.worldbank.org/en/dataset/OWID_CB)
- Coverage: 1960-2023, 205 countries
- License: CC BY

---

## Author

**Ju Ho Kim**

---

## Contact

Feel free to reach me out 😁! Thank you!
- Email: juho_kim@berkeley.edu
- [GitHub Repo](https://github.com/jurinho17-sv/global-co2-insight)
