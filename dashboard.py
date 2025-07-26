import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download model from Hugging Face Hub
model_path = hf_hub_download(repo_id="Leron7/finance", filename="savings_predictor.pkl")

# Load the model
with open(model_path, "rb") as file:
    model = joblib.load(file)

st.set_page_config(page_title="💸 Financial Guidance App", layout="wide")
st.title("💡 Personal Financial Guidance Dashboard")

st.markdown("---")

# --- User Input ---
st.header("📥 Enter Your Financial Details")

income = st.number_input("Monthly Income", min_value=0.0)
age = st.slider("Age", 18, 80, 25)
dependents = st.number_input("Number of Dependents", 0, 10, 0)
occupation = st.selectbox("Occupation", ["Salaried", "Self_Employed", "Student", "Professional", "Retired"])
city_tier = st.selectbox("City Tier", ["Tier_1", "Tier_2", "Tier_3"])

# Expenses
st.subheader("🏠 Monthly Expenses")
rent = st.number_input("Rent")
loan = st.number_input("Loan Repayment")
insurance = st.number_input("Insurance")
groceries = st.number_input("Groceries")
transport = st.number_input("Transport")
eating_out = st.number_input("Eating Out")
entertainment = st.number_input("Entertainment")
utilities = st.number_input("Utilities")
healthcare = st.number_input("Healthcare")
education = st.number_input("Education")
misc = st.number_input("Miscellaneous")

# Income Sources
st.subheader("💼 Income Portfolio")
salary = st.number_input("Salary Income")
freelance = st.number_input("Freelance Income")
investments = st.number_input("Investment Income")
other_income = st.number_input("Other Income")

# Button
if st.button("🔍 Analyze My Finances"):
    disposable_income = income - sum([
        rent, loan, insurance, groceries, transport, eating_out,
        entertainment, utilities, healthcare, education, misc
    ])
    
    df = pd.DataFrame([{
        "Income": income,
        "Age": age,
        "Dependents": dependents,
        "Occupation": occupation,
        "City_Tier": city_tier,
        "Rent": rent,
        "Loan_Repayment": loan,
        "Insurance": insurance,
        "Groceries": groceries,
        "Transport": transport,
        "Eating_Out": eating_out,
        "Entertainment": entertainment,
        "Utilities": utilities,
        "Healthcare": healthcare,
        "Education": education,
        "Miscellaneous": misc,
        "Disposable_Income": disposable_income
    }])

    predicted_savings = model.predict(df)[0]

    st.success(f"💰 **Predicted Desired Savings**: ₹{round(predicted_savings, 2)}")

    # --- Recommendations ---
    st.markdown("### ✍️ Recommendations")
    recs = []
    if income > 0 and disposable_income / income < 0.15:
        recs.append("⚠️ Low savings rate. Try reducing spending.")
    if groceries > 6000:
        recs.append("🛒 Reduce groceries.")
    if eating_out > 2000:
        recs.append("🍴 Eating out too much.")
    if transport > 2500:
        recs.append("🚌 High transport costs.")
    if entertainment > 1500:
        recs.append("🎮 Too much on entertainment.")
    if not recs:
        recs.append("✅ Spending looks healthy!")

    for r in recs:
        st.write(r)

    # --- Income Portfolio ---
    st.markdown("### 📊 Income Portfolio Breakdown")
    total_sources = salary + freelance + investments + other_income
    if total_sources > 0:
        income_breakdown = {
            "Salary": round((salary / total_sources) * 100, 2),
            "Freelance": round((freelance / total_sources) * 100, 2),
            "Investments": round((investments / total_sources) * 100, 2),
            "Other": round((other_income / total_sources) * 100, 2),
    }

        st.markdown("### 💼 Income Source Breakdown")
        st.table(pd.DataFrame(income_breakdown.items(), columns=["Source", "Percentage (%)"]))
    else:
        st.warning("No income sources entered.")

    # --- Budget Plan ---
    st.markdown("### 📅 Suggested Budget (50/30/20)")
    budget = {
        "Needs (50%)": round(income * 0.5),
        "Wants (30%)": round(income * 0.3),
        "Savings (20%)": round(income * 0.2)
    }
    st.markdown(f"**Needs (50%)**: ₹{budget['Needs (50%)']}")
    st.markdown(f"**Wants (30%)**: ₹{budget['Wants (30%)']}")
    st.markdown(f"**Savings (20%)**: ₹{budget['Savings (20%)']}")

    # --- Health Score ---
    st.markdown("### 🧠 Financial Health Score")
    savings_rate = (income - sum([rent, loan, insurance, groceries, transport, eating_out,
        entertainment, utilities, healthcare, education, misc])) / income if income >0 else 0
    dependency = salary / total_sources if total_sources > 0 else 1
    score = min((savings_rate / 0.2) * 30, 30) + (1 - dependency) * 30
    total_exp = sum([rent, loan, insurance, groceries, transport, eating_out,
                 entertainment, utilities, healthcare, education, misc])

    if income > 0:
        spending_ratio = total_exp / income
    else:
        spending_ratio = 1  # assume worst case

# Now use the safe `spending_ratio`
    if spending_ratio <= 0.5:
        score += 40
    elif spending_ratio <= 0.7:
        score += 30
    elif spending_ratio <= 0.9:
        score += 20
    else:
        score += 10
    score = round(score, 2)

    if score >= 80:
        st.success(f"✅ Score: {score} / 100 - Excellent")
    elif score >= 60:
        st.info(f"⚠️ Score: {score} / 100 - Decent")
    else:
        st.error(f"❌ Score: {score} / 100 - Needs improvement")
    # --- Investment Portfolio Analyzer ---
    st.markdown("### 📈 Recommended Investment Portfolio")

    risk_level = st.selectbox("Your Risk Appetite", ["Low", "Medium", "High"])
    
    def get_investment_allocation(income, risk_level):
        risk_level = risk_level.lower()
        if risk_level == "low":
            allocation = {"Equity": 0.2, "Debt": 0.7, "Gold": 0.1}
        elif risk_level == "medium":
            allocation = {"Equity": 0.5, "Debt": 0.4, "Gold": 0.1}
        elif risk_level == "high":
            allocation = {"Equity": 0.7, "Debt": 0.2, "Gold": 0.1}
        else:
            allocation = {"Equity": 0.5, "Debt": 0.4, "Gold": 0.1}
        
        investment_plan = {k: round(v * income, 2) for k, v in allocation.items()}
        return allocation, investment_plan

    if income > 0:
        allocation_percent, investment_amounts = get_investment_allocation(income, risk_level)

# ✅ Convert decimal to percentage for display
        allocation_percent_display = {k: round(v * 100, 2) for k, v in allocation_percent.items()}

# ✅ Format investment amounts with commas
        investment_amounts_display = {k: f"₹{round(v):,}" for k, v in investment_amounts.items()}

# 📊 Display Allocation Percentages
        st.markdown("📊 **Allocation Percentages**")
        st.table(pd.DataFrame(allocation_percent_display.items(), columns=["Asset Class", "Percentage (%)"]))

# 💸 Display Suggested Investment Amounts
        st.markdown("💸 **Suggested Investment Amounts**")
        st.table(pd.DataFrame(investment_amounts_display.items(), columns=["Asset Class", "Amount"]))
    else:
        st.warning("Please enter your income to view investment suggestions.")

