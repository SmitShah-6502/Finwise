# import streamlit as st

# st.set_page_config(page_title="AI Budget Recommender", layout="wide")
# st.title("💰 AI Budget Recommender")

# st.markdown(
#     """
#     <style>
#     @keyframes rain {
#         0% {transform: translateY(-10%);opacity: 1;}
#         100% {transform: translateY(110vh);opacity: 0;}
#     }
#     .emoji-rain {position: fixed;top: 0;left: 0;pointer-events: none;width: 100vw;height: 100vh;overflow: hidden;z-index: 9999;}
#     .emoji-rain span {position: absolute;top: -2em;font-size: 20px;animation-name: rain;animation-timing-function: linear;animation-iteration-count: infinite;animation-duration: 7s;user-select: none;}
#     .emoji-rain span:nth-child(1) {left: 10%;animation-delay: 0s;}
#     .emoji-rain span:nth-child(2) {left: 25%;animation-delay: 1.5s;animation-duration: 6s;}
#     .emoji-rain span:nth-child(3) {left: 40%;animation-delay: 3s;animation-duration: 8s;}
#     .emoji-rain span:nth-child(4) {left: 55%;animation-delay: 2s;animation-duration: 7s;}
#     .emoji-rain span:nth-child(5) {left: 70%;animation-delay: 4s;animation-duration: 6.5s;}
#     .emoji-rain span:nth-child(6) {left: 85%;animation-delay: 1s;animation-duration: 7.5s;}
#     </style>
#     <div class="emoji-rain">
#         <span>💵</span><span>💵</span><span>💵</span><span>💵</span><span>💵</span><span>💵</span>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# monthly_income = st.number_input("Enter your total monthly income (in your currency):", min_value=0.0, format="%.2f")

# goals = st.multiselect(
#     "Select your financial goals (choose one or more):",
#     options=["Retirement Savings","Emergency Fund","Debt Repayment","Vacation Fund","Home Purchase","Education Savings","Investment Growth","General Savings"]
# )

# debt_amount = 0
# if "Debt Repayment" in goals:
#     debt_amount = st.number_input("Enter your monthly debt repayment amount (optional):", min_value=0.0, format="%.2f")

# if st.button("Generate Budget Recommendation"):
#     if monthly_income <= 0:
#         st.warning("Please enter a valid monthly income greater than 0.")
#     else:
#         essentials_pct, savings_pct, discretionary_pct, debt_pct = 50, 30, 20, 0

#         if "Debt Repayment" in goals and debt_amount > 0:
#             debt_pct = (debt_amount / monthly_income) * 100
#             discretionary_pct -= debt_pct
#             if discretionary_pct < 5: discretionary_pct = 5
#             savings_pct = 100 - essentials_pct - debt_pct - discretionary_pct

#         if "Retirement Savings" in goals:
#             savings_pct += 10; discretionary_pct -= 10
#         if "Emergency Fund" in goals:
#             savings_pct += 5; discretionary_pct -= 5

#         if discretionary_pct < 5:
#             discretionary_pct = 5
#             savings_pct = 100 - essentials_pct - debt_pct - discretionary_pct

#         st.markdown(f"### Budget Allocation based on your inputs:")
#         st.write(f"- Essentials (housing, food, utilities): **{essentials_pct:.1f}%**")
#         st.write(f"- Savings (retirement, emergency, investment): **{savings_pct:.1f}%**")
#         if debt_pct > 0: st.write(f"- Debt Repayment: **{debt_pct:.1f}%**")
#         st.write(f"- Discretionary Spending (leisure, shopping): **{discretionary_pct:.1f}%**")

#         st.markdown("### Monthly Amounts:")
#         st.write(f"- Essentials: {monthly_income * essentials_pct / 100:.2f}")
#         st.write(f"- Savings: {monthly_income * savings_pct / 100:.2f}")
#         if debt_pct > 0: st.write(f"- Debt Repayment: {monthly_income * debt_pct / 100:.2f}")
#         st.write(f"- Discretionary: {monthly_income * discretionary_pct / 100:.2f}")

#         st.info("This is a basic recommendation. For personalized advice, consider consulting a financial advisor.")



import streamlit as st

st.set_page_config(page_title="AI Budget Recommender", layout="wide")
st.title("💰 AI Budget Recommender")

st.markdown(
    """
    <style>
    @keyframes rain {
        0% {transform: translateY(-10%);opacity: 1;}
        100% {transform: translateY(110vh);opacity: 0;}
    }
    .emoji-rain {position: fixed;top: 0;left: 0;pointer-events: none;width: 100vw;height: 100vh;overflow: hidden;z-index: 9999;}
    .emoji-rain span {position: absolute;top: -2em;font-size: 20px;animation-name: rain;animation-timing-function: linear;animation-iteration-count: infinite;animation-duration: 7s;user-select: none;}
    .emoji-rain span:nth-child(1) {left: 10%;animation-delay: 0s;}
    .emoji-rain span:nth-child(2) {left: 25%;animation-delay: 1.5s;animation-duration: 6s;}
    .emoji-rain span:nth-child(3) {left: 40%;animation-delay: 3s;animation-duration: 8s;}
    .emoji-rain span:nth-child(4) {left: 55%;animation-delay: 2s;animation-duration: 7s;}
    .emoji-rain span:nth-child(5) {left: 70%;animation-delay: 4s;animation-duration: 6.5s;}
    .emoji-rain span:nth-child(6) {left: 85%;animation-delay: 1s;animation-duration: 7.5s;}
    </style>
    <div class="emoji-rain">
        <span>💵</span><span>💵</span><span>💵</span><span>💵</span><span>💵</span><span>💵</span>
    </div>
    """,
    unsafe_allow_html=True,
)

monthly_income = st.number_input("Enter your total monthly income (in your currency):", min_value=0.0, format="%.2f")

goals = st.multiselect(
    "Select your financial goals (choose one or more):",
    options=["Retirement Savings","Emergency Fund","Debt Repayment","Vacation Fund","Home Purchase","Education Savings","Investment Growth","General Savings"]
)

debt_amount = 0
if "Debt Repayment" in goals:
    debt_amount = st.number_input("Enter your monthly debt repayment amount (optional):", min_value=0.0, format="%.2f")

if st.button("Generate Budget Recommendation"):
    if monthly_income <= 0:
        st.warning("Please enter a valid monthly income greater than 0.")
    else:
        essentials_pct, savings_pct, discretionary_pct, debt_pct = 50, 30, 20, 0

        if "Debt Repayment" in goals and debt_amount > 0:
            debt_pct = (debt_amount / monthly_income) * 100
            discretionary_pct -= debt_pct
            if discretionary_pct < 5: discretionary_pct = 5
            savings_pct = 100 - essentials_pct - debt_pct - discretionary_pct

        if "Retirement Savings" in goals:
            savings_pct += 10; discretionary_pct -= 10
        if "Emergency Fund" in goals:
            savings_pct += 5; discretionary_pct -= 5

        if discretionary_pct < 5:
            discretionary_pct = 5
            savings_pct = 100 - essentials_pct - debt_pct - discretionary_pct

        st.markdown(f"### Budget Allocation based on your inputs:")
        st.write(f"- Essentials (housing, food, utilities): **{essentials_pct:.1f}%**")
        st.write(f"- Savings (retirement, emergency, investment): **{savings_pct:.1f}%**")
        if debt_pct > 0: st.write(f"- Debt Repayment: **{debt_pct:.1f}%**")
        st.write(f"- Discretionary Spending (leisure, shopping): **{discretionary_pct:.1f}%**")

        st.markdown("### Monthly Amounts:")
        st.write(f"- Essentials: {monthly_income * essentials_pct / 100:.2f}")
        st.write(f"- Savings: {monthly_income * savings_pct / 100:.2f}")
        if debt_pct > 0: st.write(f"- Debt Repayment: {monthly_income * debt_pct / 100:.2f}")
        st.write(f"- Discretionary: {monthly_income * discretionary_pct / 100:.2f}")

        st.info("This is a basic recommendation. For personalized advice, consider consulting a financial advisor.")
