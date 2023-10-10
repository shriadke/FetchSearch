import streamlit as st
from fetchSearch.pipeline.prediction import PredictionPipeline

st.title('Welcome to Fetch Offers App')
st.title('🔎 _Search_ for :blue[offers] across our platform!  :sunglasses:')
st.markdown("""Example Search Options (_but not limited to_):""") 
st.markdown("""1. category (ex. dairy)\n2. Brand name (ex. Beyond meat)\n3. Retailer name (ex.Walmart)""")
st.markdown("""The scores  shown are the relevance percentage obtained from the semantic search model! :astonished:""")

with st.form("search_form"):
	text = st.text_input("Search for offers below:", placeholder = "Search something and Press search")
	submit_status = st.form_submit_button("Search")
	if submit_status:
		pred_obj = PredictionPipeline()

		output_df = pred_obj.predict(text)
		st.dataframe(output_df,hide_index=True)
      