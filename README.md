# Paper-Trading Bot

A comprehensive, fully automated paper trading and backtesting framework built in Python. This bot is designed to evaluate market conditions, generate trading signals, and simulate risk-free order execution by combining technical indicators, fundamental screening, and pattern recognition. 

## Key Features & Capabilities

Based on the core engine files, this repository supports a highly modular trading workflow:

* **Dual-Analysis Engine:** Combines fundamental asset screening (`fundamental_screener.py`) with technical market analysis (`technical_indicators.py`).
* **Algorithmic Signal Generation:** Utilizes an `alpha_engine.py` and a dedicated `signal_generator.py` to identify optimal entry and exit points. 
* **Pattern Recognition:** Leverages weighted historical patterns or machine learning inputs via the `pattern_weights.csv` file to validate trade setups.
* **Robust Trade Management:** The `paper_trading_manager.py` handles mock order execution, while trade histories and portfolio balances are tracked locally in `paper_trades.csv` and `daily_equity.csv`.
* **Automated Workflows:** Includes GitHub Actions integration (`.github/workflows/main.yml`) for cloud-based automation and a `dailytask.bat` script for local scheduled execution.
* **Real-time Alerts:** Integrated `notification_handler.py` to broadcast live trade executions, signals, and system alerts.
* **Historical Backtesting:** Features a dedicated `backtest_analytics.py` script to rigorously evaluate trading strategies against past data before paper trading.

---

## Repository Structure

The codebase is organized into distinct execution scripts, core logic modules, and comprehensive test suites:

### Execution Scripts
These are the primary entry points for running the bot:
* `run_paper_trading.py`: Initiates the live paper trading simulation.
* `run_live_screening.py`: Runs real-time market scans to find assets matching the active criteria.
* `run_backtest.py`: Executes the strategy against historical data to evaluate performance.
* `swing_trading_bot.py`: The overarching logic tying the swing-trading ruleset together.

### Core Logic & Engines
The modular components that drive decision-making:
* `alpha_engine.py`: Core alpha-generating logic.
* `signal_generator.py`: Converts indicators and data into actionable buy/sell triggers.
* `technical_indicators.py`: Mathematical functions for market momentum and trend analysis.
* `fundamental_screener.py`: Filters assets based on underlying financial metrics.
* `paper_trading_manager.py`: Manages the simulated portfolio state.
* `data_fetcher_free.py`: Ingests market pricing data using free-tier API services.

### Testing Suite
A full suite of unit tests ensuring stability and reliable logic execution:
* `test_data_fetcher_free.py`
* `test_fundamental_screener.py`
* `test_notification_handler.py`
* `test_signal_generator.py`
* `test_swing_trading_bot.py`
* `test_technical_indicators.py`

---

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies outlined in the project configuration:
```bash
git clone https://github.com/tanmaykaper/Paper-Trading-Bot.git
cd Paper-Trading-Bot
pip install -r requirements.txt
```

### 2. Usage Execution
Depending on your objective, run one of the primary execution files:

**To run a historical backtest:**
```bash
python run_backtest.py
```

**To screen the market for live setups:**
```bash
python run_live_screening.py
```

**To start the live paper-trading simulation:**
```bash
python run_paper_trading.py
```

### 3. Automation
* **Local:** Schedule `dailytask.bat` via Windows Task Scheduler or cron.
* **Cloud:** The `.github/workflows/main.yml` file allows the bot to run automatically on a set schedule using GitHub Actions.

---

## Disclaimer
*This software is intended strictly for educational, research, and simulation purposes. It does not constitute financial or investment advice. Simulated paper trading results do not guarantee future performance in live markets.*
