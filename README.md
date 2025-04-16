# Financial Dashboard

## Project Overview

An interactive financial dashboard designed to analyze stock market data and individual companies. Built with Streamlit, the app integrates real-time market data and provides tools for single stock analysis, company comparison, portfolio evaluation, and monitoring of key financial indicators.

## Features

- **Single Stock Analysis**  
  Users can select a company from a dropdown list, choose a time period and data interval. The app fetches historical market data, displays price charts, and calculates key technical indicators.

- **Company Comparison**  
  Compare multiple companies based on selected metrics such as price change, trading volume, or technical indicators.

- **Portfolio Analysis**  
  Analyze a custom investment portfolio composed of selected companies. Performance is evaluated over time, including individual stock weights and historical returns.

- **Market Indicators**  
  View analyst recommendations for selected stocks and visualize them through charts and tables.

## Technologies Used

- Python  
- Streamlit  
- Pandas  
- yfinance  
- matplotlib / plotly

## Requirements

To run the project locally, the following libraries are required (listed in `requirements.txt`):

- streamlit  
- yfinance  
- pandas  
- matplotlib  
- plotly  
- numpy

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the app:

```bash
streamlit run app.py
```

## Project Status

The project is currently in development. Future updates may include additional indicators, API integrations, and expanded portfolio analysis capabilities.
