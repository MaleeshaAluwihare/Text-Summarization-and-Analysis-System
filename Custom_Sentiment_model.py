from transformers import BertModel
import torch.nn as nn

class BertForMultiTask(nn.Module):
    def __init__(self, bert_model_name, num_sentiment_labels, num_article_labels):
        super(BertForMultiTask, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, num_sentiment_labels)
        self.article_classifier = nn.Linear(self.bert.config.hidden_size, num_article_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        sentiment_logits = self.sentiment_classifier(pooled_output)
        article_logits = self.article_classifier(pooled_output)
        return sentiment_logits, article_logits

# Initialize the custom model
model = BertForMultiTask('bert-base-uncased', num_sentiment_labels=3, num_article_labels=7)