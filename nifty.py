#!/usr/bin/env python
# coding: utf-8

# In[6]:


from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


# In[9]:


model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")


# In[10]:


nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# In[13]:


sentences = ["Technopolis plans to develop in stages an area of no less than 100,000 square meters in order to host companies working in computer technologies and telecommunications , the statement said .",
             "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .",
             "With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability .",
             "A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser .",
             "The personnel reductions will primarily affect those working for the parent company in the diagnostics business or in production and logistics in the liquid handling business .",
             "Sensex plunges 700 points as weak economic data spooks investors; Nifty below 19,550"]


# In[14]:


results = nlp(sentences)
print(results)


# In[15]:


model = BertForSequenceClassification.from_pretrained("FinancialBERT",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("FinancialBERT")


# In[18]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Set the HF_HOME environment variable with your token
os.environ['HF_HOME'] = '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'

# Replace "your_username/your_private_model" with the actual model name
model_name = "suhani112/FinancialBERT"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("FinancialBERT")
model = AutoModelForSequenceClassification.from_pretrained("FinancialBERT")

# Now you can use the loaded tokenizer and model for your NLP tasks


# In[19]:


model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")


# In[20]:


nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# In[21]:


sentences = ["Technopolis plans to develop in stages an area of no less than 100,000 square meters in order to host companies working in computer technologies and telecommunications , the statement said .",
             "The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .",
             "With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability .",
             "A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser .",
             "The personnel reductions will primarily affect those working for the parent company in the diagnostics business or in production and logistics in the liquid handling business .",
             "Sensex plunges 700 points as weak economic data spooks investors; Nifty below 19,550"]


# In[22]:


results = nlp(sentences)
print(results)


# In[ ]:




