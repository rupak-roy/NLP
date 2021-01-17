# Core Pkgs
import streamlit as st
import streamlit.components.v1 as stc


# Additional Pkgs
# Load EDA Pkgs
import pandas as pd

#------------------------------------------------

#-----------------------------------------------------
# Text Cleaning Pkgs
import neattext as nt
import neattext.functions as nfx 

# utils

import base64
import time

timestr = time.strftime("%Y%m%d-%H%M%S")


# Data Viz Pkgs
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# External Utils
from app_utils import *
import nlp_utils


# Fxn to Get Wordcloud
from wordcloud import WordCloud


def plot_wordcloud(my_text):
    my_wordcloud = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(my_wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)


# Fxn to Download Result
def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "Analysis_report_{}_.csv".format(timestr)
    st.markdown("🤘🏻  Download CSV file ⬇️  ")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)


# Fxn for LexRank Summarization
# Function for Sumy Summarization
def sumy_summarizer(docx,num=2):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,num)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result
#------------------TEXT SUMARIZATION PKGS-------------------------
# Additional Pkgs /Summarization Pkgs
# TextRank Algorithm
from gensim.summarization import summarize 

# LexRank Algorithm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# import seaborn as sns 
import altair as alt 


# Evaluate Summary
from rouge import Rouge 
def evaluate_summary(summary,reference):
	r = Rouge()
	eval_score = r.get_scores(summary,reference)
	eval_score_df = pd.DataFrame(eval_score[0])	
	return eval_score_df

#----------------------------------------------------------------

def plot_mendelhall_curve(docx):
	word_length = [ len(token) for token in docx.split()]
	word_length_count = Counter(word_length)
	sorted_word_length_count = sorted(dict(word_length_count).items())
	x,y = zip(*sorted_word_length_count)
	fig = plt.figure(figsize=(20,10))
	plt.plot(x,y)
	plt.title("Plot of Word Length Distribution")
	plt.show()
	st.pyplot(fig)


def plot_mendelhall_curve_2(docx):
	word_length = [ len(token) for token in docx.split()]
	word_length_count = Counter(word_length)
	sorted_word_length_count = sorted(dict(word_length_count).items())
	x,y = zip(*sorted_word_length_count)
	mendelhall_df = pd.DataFrame({'tokens':x,'counts':y})
	st.line_chart(mendelhall_df['counts'])


st.image('img/nlp.jpg', width=720)
#st.color_picker("Pick a Theme color")

def main():
    st.title("Text Analysis NLP _Beta v1.0")
    menu = ["Home", "Upload", "About"]

    choice = st.sidebar.selectbox("NLP Menu", menu)
    if choice == "Home":
        st.write("Our day to day language can tell you an aboard patterns, insights and sentiments. Explore the prower of Ai: Natural Language Processing algorithim and discover synchronicity that leads one to another. Free to use as much as you like! under GNU General Public License with a Motto #WeRiseByLiftingOthers")
        st.write("Sample Dataset [@rupak-roy Github](https://github.com/rupak-roy/dataset-streamlit) V3 update: Deep Learning module at https://share.streamlit.io/rupak-roy/streamlit_deeplearning_analytics/main/ML.py")
        raw_text = st.text_area("Enter Text Here")
        num_of_most_common = st.sidebar.number_input("Min Common Keywords", 5, 15)
        if st.button("Analyze"):

     #       with st.beta_expander("Original Text"):
     #          st.write(raw_text)
            with st.beta_expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)
                
            with st.beta_expander("Entities Explorer"):
                # entity_result = get_entities(raw_text)
                # st.write(entity_result)

                entity_result = render_entities(raw_text)
                stc.html(entity_result, height=300, scrolling=True)
                
                
            with st.beta_expander("Summary using LexRank Approach"):
                st.text("Disclaimer: LexRank is an unsupervised approach to text summarization based on graph-based centrality scoring of sentences. The main idea is that sentences “recommend” other similar sentences to the reader. Thus, if one sentence is very similar to many others, it will likely be a sentence of great importance. The importance of this sentence also stems from the importance of the sentences “recommending” it. Thus, to get ranked highly and placed in a summary, a sentence must be similar to many sentences that are in turn also similar to many other sentences. This makes intuitive sense and allows the algorithms to be applied to any arbitrary new text.")
                my_summary = sumy_summarizer(raw_text)
                document_len = {"Original":len(raw_text),
				"Summary":len(my_summary)}
                st.write(document_len)
                st.write(my_summary)
                        
                st.info("Rouge Score: F-Score:The Higher the Better the results, R-Recall/Sensitivty: refers correctly predicted positive observations to the all observations was actually positive, P - Precision talks about how precise/accurate your model is out of those predicted positive, how many of them are actual positive. Ranges from 0-1 The higher the score the better the model is.")
                eval_df= evaluate_summary(my_summary,raw_text)
                st.dataframe(eval_df.T)
                eval_df['metrics'] = eval_df.index
                c = alt.Chart(eval_df).mark_bar().encode(
    					x='metrics',y='rouge-1')
                st.altair_chart(c)

            with st.beta_expander("Summary using TextRank Approach"):
                st.text("Note: One of the famous Text Summarization algorithm gets its name from Larry Page, one of the co-founders of Google.")
                my_summary = summarize(raw_text)
                document_len = {"Original":len(raw_text),
                "Summary":len(my_summary)}
                st.write(document_len)
                st.write(my_summary)
                st.info("Rouge Score: F-Score:The Higher the Better the results, R-Recall/Sensitivty: refers correctly predicted positive observations to the all observations was actually positive, P - Precision talks about how precise/accurate your model is out of those predicted positive, how many of them are actual positive. Ranges from 0-1 The higher the score the better the model is.")
                eval_df = evaluate_summary(my_summary,raw_text)
                st.dataframe(eval_df)
                eval_df['metrics'] = eval_df.index
                c = alt.Chart(eval_df).mark_bar().encode(
					x='metrics',y='rouge-1')
                st.altair_chart(c)




            # Layouts
            col1, col2 = st.beta_columns(2)

            with col1:

                    
                    
                with st.beta_expander("Word Statistics"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.beta_expander("Top Keywords/Tokens"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    st.write(keywords)

                with st.beta_expander("Sentiment Explorer"):
                    st.info("Sentiment Analysis")
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)

            with col2:

                    
                with st.beta_expander("Word Frequency Graph"):
                    fig = plt.figure()
                    top_keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    plt.bar(keywords.keys(), top_keywords.values())
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.beta_expander("Part of Speech(PoS) Graph"):
                	try:
                		fig = plt.figure()
                		sns.countplot(token_result_df["PoS"])
                		plt.xticks(rotation=45)
                		st.pyplot(fig)
                	except:
                		st.warning("Error: Insufficient Data")

                with st.beta_expander("Plot Wordcloud"):
                    try:
                    	plot_wordcloud(raw_text)
                    except:
                    	st.warning("Error: Insufficient Data")
                
                with st.beta_expander("Stylography Explorer"):
                    st.info("using Mendelhall Curve")
                    plot_mendelhall_curve_2(raw_text)
                
                
                
            with st.beta_expander("Download The Analysis Report"):
                make_downloadable(token_result_df)

    elif choice == "Upload":
        st.write("Our day to day language can tell you an aboard patterns, insights and sentiments. Explore the prower of Ai: Natural Language Processing algorithim and discover synchronicity that leads one to another. Free to use as much as you like! under GNU General Public License with a Motto #WeRiseByLiftingOthers")

        text_file = st.file_uploader("Upload Files", type=["pdf", "docx", "txt"])
        num_of_most_common = st.sidebar.number_input("Min Common Keywords", 5, 15)

        if text_file is not None:
            if text_file.type == "application/pdf":
                raw_text = read_pdf(text_file)
                # st.write(raw_text)
            elif text_file.type == "text/plain":
                # st.write(text_file.read()) # read as bytes
                raw_text = str(text_file.read(), "utf-8")
                # st.write(raw_text)
            else:
                raw_text = docx2txt.process(text_file)
                # st.write(raw_text)

            with st.beta_expander("Original Text"):
                st.write(raw_text)

            with st.beta_expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)

            with st.beta_expander("Entities Explorer"):
                # entity_result = get_entities(raw_text)
                # st.write(entity_result)

                entity_result = render_entities(raw_text)
                stc.html(entity_result, height=300, scrolling=True)
                
            with st.beta_expander("Summary using LexRank Approach"):
                st.text("Disclaimer: LexRank is an unsupervised approach to text summarization based on graph-based centrality scoring of sentences. The main idea is that sentences “recommend” other similar sentences to the reader. Thus, if one sentence is very similar to many others, it will likely be a sentence of great importance. The importance of this sentence also stems from the importance of the sentences “recommending” it. Thus, to get ranked highly and placed in a summary, a sentence must be similar to many sentences that are in turn also similar to many other sentences. This makes intuitive sense and allows the algorithms to be applied to any arbitrary new text.")
                my_summary = sumy_summarizer(raw_text)
                document_len = {"Original":len(raw_text),
				"Summary":len(my_summary)}
                st.write(document_len)
                st.write(my_summary)
                        
                st.info("Rouge Score: F-Score:The Higher the Better the results, R-Recall/Sensitivty: refers correctly predicted positive observations to the all observations was actually positive, P - Precision talks about how precise/accurate your model is out of those predicted positive, how many of them are actual positive. Ranges from 0-1 The higher the score the better the model is.")
                eval_df= evaluate_summary(my_summary,raw_text)
                st.dataframe(eval_df.T)
                eval_df['metrics'] = eval_df.index
                c = alt.Chart(eval_df).mark_bar().encode(
    					x='metrics',y='rouge-1')
                st.altair_chart(c)

            with st.beta_expander("Summary using TextRank Approach"):
                st.text("Note: One of the famous Text Summarization algorithm gets its name from Larry Page, one of the co-founders of Google.")
                my_summary = summarize(raw_text)
                document_len = {"Original":len(raw_text),
                "Summary":len(my_summary)}
                st.write(document_len)
                st.write(my_summary)
                
                st.info("Rouge Score: F-Score:The Higher the Better the results, R-Recall/Sensitivty: refers correctly predicted positive observations to the all observations was actually positive, P - Precision talks about how precise/accurate your model is out of those predicted positive, how many of them are actual positive. Ranges from 0-1 The higher the score the better the model is.")
                eval_df = evaluate_summary(my_summary,raw_text)
                st.dataframe(eval_df)
                eval_df['metrics'] = eval_df.index
                c = alt.Chart(eval_df).mark_bar().encode(
					x='metrics',y='rouge-1')
                st.altair_chart(c)     
                
                

            # Layouts
            col1, col2 = st.beta_columns(2)

            with col1:
                with st.beta_expander("Word Statistics"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.beta_expander("Top Keywords/Tokens"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    st.write(keywords)

                with st.beta_expander("Sentiment Explorer"):
                    st.info("Sentiment Analysis")
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)

            with col2:
                with st.beta_expander("Word Frequency Graph"):
                    fig = plt.figure()
                    top_keywords = get_most_common_tokens(
                        processed_text, num_of_most_common
                    )
                    plt.bar(keywords.keys(), top_keywords.values())
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with st.beta_expander("Part of Speech(Pos) Graph"):
                    try:

                        fig = plt.figure()
                        sns.countplot(token_result_df["PoS"])
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    except:
                        st.warning("Error: Insufficient Data")

                with st.beta_expander("Plot Wordcloud"):
                	try:
                		plot_wordcloud(raw_text)
                	except:
                		st.warning("Error: Insufficient Data")
                        
                with st.beta_expander("Stylography Explorer"):
                    st.info("using Mendelhall Curve")
                    plot_mendelhall_curve_2(raw_text)
                    
            if st.sidebar.checkbox("Top Keywords/Tokens"):
                st.info("Top Keywords/Tokens")
                processed_text = nfx.remove_stopwords(raw_text)
                keywords = get_most_common_tokens(
                    processed_text, num_of_most_common
                )
                st.write(keywords)
            
            if st.sidebar.checkbox("Part of Speech(Pos) Graph"):
                fig = plt.figure()
                sns.countplot(token_result_df["PoS"])
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            if st.sidebar.checkbox("Sentiment Analysis"):
                st.info("Sentiment Analysis")
                sent_result = get_sentiment(raw_text)
                st.write(sent_result)
                
            if st.sidebar.checkbox("Stylography Analysis"):
                st.info("using Mendelhall Curve")
                plot_mendelhall_curve_2(raw_text)
                
                                
            if st.sidebar.checkbox("Plot Word Frequency Graph"):
                fig = plt.figure()
                top_keywords = get_most_common_tokens(
                    processed_text, num_of_most_common
                )
                plt.bar(keywords.keys(), top_keywords.values())
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
            if st.sidebar.checkbox("Plot WordCloud"):
                plot_wordcloud(raw_text)
                
            with st.beta_expander("Download The Analysis Report"):
                make_downloadable(token_result_df)

    else:
        st.subheader("About")
        st.text("Thank you for your time")
        
        st.markdown("""
Hi I’m Bob aka. Rupak Roy. Things i write about frequently on Quora & Linkedin: analytics For Beginners, Data Science, Machine Learning, Deep learning, Natural Language Processing (NLP), Computer Vision, Big Data Technologies, Internet Of Thins and many other random topics of interest.
I formerly Co-founded various Ai based projects to inspire and nurture the human spirit with the Ai training on how to leverage on how to leverage Ai to solve problems for an exponential growth.

My Career Contour consists of various technologies starting from Masters of Science in Information Technology to Commerce with the privilege to be Wiley certified in various Analytical Domain. My alternative internet presences, Facebook, Blogger, Linkedin, Medium, Instagram, ISSUU and with Data2Dimensions
If you wish to learn more about Data Science follow me at:

~ Medium [@rupak.roy](https://medium.com/@rupak.roy)

~ Linkedin [@bobrupak](https://www.linkedin.com/in/bobrupak/)

My Fav. Quote:

Millions saw the apple fall but only Newton asked why! ~ “Curiosity is the spark of perfection and innovations. So connect with data and discover sync“
""")
        st.image('img/prism.gif')
        with st.beta_expander("Suprise!"):
            st.title("COLLECT YOUR FULL VERSION MACHINE LEARNING APP @ ping_me #socialmedia")
            st.image('img/office.jpg')
            st.info("")
            st.success("")
            st.warning("")
            st.error("")

            
            
        
            


if __name__ == "__main__":
    main()
