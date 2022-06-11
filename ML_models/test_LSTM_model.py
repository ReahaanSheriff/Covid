import unittest
import LSTM_model

class TestLSTM(unittest.TestCase):
    def test_lstmForecasting(self):
        first = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/lstmtrain.png"
        second = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/lstmpredict.png"
        self.assertListEqual(LSTM_model.lstmForecasting('WHO-COVID-19-global-data.csv'), 
        [first,second,71269.10871074298,6592136520.412533,81191.97325113199])