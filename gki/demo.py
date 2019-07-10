import numpy as np
import tensorflow as tf
import preprocess
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.Qt import QStringListModel, QModelIndex
from qwikidata.linked_data_interface import get_entity_dict_from_api

def run_ui():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())

class demo:
    def __init__(self, sq=preprocess.SimpleQuestions()):
        self.pre = sq
        self.knowledgebase = sq.knowledgebase
        self.words_dict = sq.words_dict
        self.knowledgebase_dict = sq.knowledgebase_dict
        
        self.w_words = tf.Variable(tf.random_normal([len(self.words_dict), 128]), name="w_words")
        self.w_entities = tf.Variable(tf.random_normal([len(self.knowledgebase_dict), 128]), name="w_entities")
        self.lstm_fw = tf.nn.rnn_cell.LSTMCell(64, name="lstm_fw", reuse=tf.AUTO_REUSE)
        self.lstm_bw = tf.nn.rnn_cell.LSTMCell(64, name="lstm_bw", reuse=tf.AUTO_REUSE)
        self.w_att = tf.Variable(tf.random_normal([128*2, 1]), name="w_att")
        self.b_att = tf.Variable(0.1, dtype=tf.float32, name="b_att")
        
        self.q_place = tf.placeholder(tf.int32,[20])
        self.c_place = tf.placeholder(tf.int32,[3])
        
        embed_q = tf.nn.embedding_lookup(self.w_words, [self.q_place]) # 1, 20, 128
        embed_c = tf.nn.embedding_lookup(self.w_entities, [self.c_place]) # 1, 3, 128
        un_q = tf.unstack(embed_q, 20, axis=1) # 20, 1, 128
        un_c = tf.unstack(embed_c, 3, axis=1) # 3, 1, 128
        t_lstm, _, _ = tf.contrib.rnn.static_bidirectional_rnn(self.lstm_fw, self.lstm_bw, un_q, dtype=tf.float32) # 20, 1, 128
        e_s = un_c[0] # 1, 128
        e_p = un_c[1]
        e_o = un_c[2]
        q_ss = []
        q_ps = []
        q_os = []
        for i in range(20):
            w_is = tf.matmul(tf.tanh(tf.concat([t_lstm[i],e_s], axis=1)), self.w_att, transpose_b=False) + self.b_att
            w_ip = tf.matmul(tf.tanh(tf.concat([t_lstm[i],e_p], axis=1)), self.w_att, transpose_b=False) + self.b_att
            w_io = tf.matmul(tf.tanh(tf.concat([t_lstm[i],e_o], axis=1)), self.w_att, transpose_b=False) + self.b_att
            # 关键的权重a的计算，
            a_is = tf.exp(w_is) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
            a_ip = tf.exp(w_ip) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
            a_io = tf.exp(w_io) / (tf.exp(w_is) + tf.exp(w_ip) + tf.exp(w_io))
            q_is = tf.nn.relu(a_is * t_lstm[i])
            q_ip = tf.nn.relu(a_ip * t_lstm[i])
            q_io = tf.nn.relu(a_io * t_lstm[i])
            q_ss.append(q_is)
            q_ps.append(q_ip)
            q_os.append(q_io)
        
        q_s = q_ss[0]
        q_p = q_ps[0]
        q_o = q_os[0]
        for i in range(1, 20):
            q_s = q_s + q_ss[i]
            q_p = q_p + q_ps[i]
            q_o = q_o + q_os[i]
        
        self.score = tf.matmul(q_s, e_s, transpose_b=True) + tf.matmul(q_p, e_p, transpose_b=True) + tf.matmul(q_o, e_o, transpose_b=True)
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "./qa_gki")
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k,v in zip(variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)
    
    def __del__(self):
        self.sess.close()
    
    def qa(self, question, subject):
        q, c = self.question_process(question, subject)
        return self.run_sess(q,c)
    
    def question_process(self, question, subject):
        question = question.lower()
        question = question.rstrip("?")
        if question[0:5]=="name ":
            question = question.rstrip(".")
        if question[0:7]=="what's ":
            question = question.replace("what's", "what is", 1)
        if question[0:6]=="who's ":
            question = question.replace("who's", "who is", 1)
        if "(" in question and ")" in question:
            question = question.replace("(", " ").replace(")", " ")
        if "http:" not in question and "https:" not in question:
            question = question.replace("/", " / ").replace("?", " ? ")
        if "\'s " in question:
            question = question.replace("\'s ", " \'s ")
        question = question.replace("\"", " ")
        question = question.replace(",", " , ").replace(":", " : ").replace("!", " ! ")
        words = question.split()
        # 忽略长度大于20的问题
        if len(words) > 20:
            return "out", "out"
        o_question = []
        for word in words:
            try:
                o_question.append(self.words_dict.index(word))
            except ValueError:
                o_question.append(0)
        while len(o_question) < 20:
            o_question.append(-1)
        
        candidate = []
        for spo in self.knowledgebase:
            if subject in spo:
                c = []
                for e in spo:
                    try:
                        c.append(self.knowledgebase_dict.index(e))
                    except ValueError:
                        c.append(0)
                candidate.append(np.array(c))
        
        return np.array(o_question), np.array(candidate)
    
    def run_sess(self, question, candidate):
        if question == "out":
            return "out of length"
        if len(candidate) == 0:
            return "no answer"
        
        scores = []
        for c in candidate:
            r_s = self.sess.run(self.score, feed_dict={self.q_place:question, self.c_place:c})
            scores.append(r_s[0][0])
        
        highest = max(scores)
        right_answer = []
        right_score = []
        result = []
        for i in range(len(scores)):
            if scores[i] > highest - 0.6:
                right_answer.append(candidate[i])
                right_score.append(scores[i])
        if len(right_score) > 5:
            for i in range(5):
                highest = max(right_score)
                result.append([right_score[right_score.index(highest)], right_answer[right_score.index(highest)]])
                del right_answer[right_score.index(highest)]
                del right_score[right_score.index(highest)]
        else:
            for i in range(len(right_score)):
                highest = max(right_score)
                result.append([right_score[right_score.index(highest)], right_answer[right_score.index(highest)]])
                del right_answer[right_score.index(highest)]
                del right_score[right_score.index(highest)]
        
        return result


class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.demo = demo()

    def init_ui(self):
        self.setFixedSize(1280,720)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout) # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget() # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout) # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget,0,0,15,4) # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget,0,4,15,12) # 右侧部件在第0行第3列，占8行9列
        self.setCentralWidget(self.main_widget) # 设置窗口主部件
        
        self.left_close = QtWidgets.QPushButton("") # 关闭按钮
        self.left_visit = QtWidgets.QPushButton("") # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮
        
        self.left_label = QtWidgets.QLabel("推荐主题")
        self.left_label.setObjectName('left_label')
        self.left_view = QtWidgets.QListView()
        self.left_view.setObjectName('left_view')
        
        self.left_layout.addWidget(self.left_close, 0,0, 1,1)
        self.left_layout.addWidget(self.left_visit, 0,1, 1,1)
        self.left_layout.addWidget(self.left_mini, 0,2, 1,1)
        self.left_layout.addWidget(self.left_label, 1,0, 1,4)
        self.left_layout.addWidget(self.left_view, 2,0, 13,4)
        
        self.right_bar_search_input = QtWidgets.QLineEdit()
        self.right_bar_search_input.setPlaceholderText("请输入有关主题，长度小于20的问题，且无特殊字符")
        
        self.right_label_title = QtWidgets.QLabel("知识库问答")
        self.right_label_title.setObjectName('right_title')
        
        self.right_layout.addWidget(self.right_label_title,0,0,1,12)
        self.right_layout.addWidget(self.right_bar_search_input,1,0,1,12)
        
        self.right_label_detail = QtWidgets.QLabel("详情")
        self.right_label_detail.setObjectName('right_label')
        self.right_label_result = QtWidgets.QLabel("结果")
        self.right_label_result.setObjectName('right_label')
        self.right_label_subject = QtWidgets.QLabel("当前主题: \tQ1")
        self.right_label_subject.setObjectName('right_label')
        
        self.right_text_detail = QtWidgets.QTextBrowser()
        self.right_text_detail.setObjectName('right_text')
        self.right_view_result = QtWidgets.QListView()
        self.right_view_result.setObjectName('right_view')
        
        self.right_layout.addWidget(self.right_label_detail,2,0,1,1)
        self.right_layout.addWidget(self.right_label_subject,2,1,1,3)
        self.right_layout.addWidget(self.right_text_detail,3,0,8,12)
        self.right_layout.addWidget(self.right_label_result,11,0,1,1)
        self.right_layout.addWidget(self.right_view_result,12,0,4,12)
        
        self.left_close.setFixedSize(15,15) # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15) # 设置最小化按钮大小
        self.left_close.setStyleSheet('''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet('''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        
        self.left_widget.setStyleSheet('''
            QLabel#left_label{
                border:none;
                color:white;
                border-bottom:1px solid white;
                font-size:20px;
                font-weight:500;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;}
            QWidget#left_widget{
                background:Gray;
                border-top:1px solid white;
                border-bottom:1px solid white;
                border-left:1px solid white;
                border-top-left-radius:10px;
                border-bottom-left-radius:10px;}
        ''')
        self.right_widget.setStyleSheet('''
            QLineEdit{
                border:1px solid gray;
                width:300px;
                border-radius:10px;
                padding:4px 4px;
                font-size:18px;
                font-weight:400;
                font-family:"Helvetica Neue", Helvetica, Arial, sans-serif;}
            QWidget#right_widget{
                background:LightGray;
                border-top:1px solid white;
                border-bottom:1px solid white;
                border-right:1px solid white;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;}
            QLabel#right_label{
                border:none;
                font-size:20px;
                font-weight:400;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;}
            QTextBrowser#right_text{
                font-size:18px;
                font-weight:400;
                font-family:"Helvetica Neue", Helvetica, Arial, sans-serif;}
            QLabel#right_title{
                border:none;
                font-size:30px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;}
        ''')
        self.left_view.setStyleSheet('''
            QListView#left_view{
                font-size:18px;
                font-weight:400;
                font-family:"Helvetica Neue", Helvetica, Arial, sans-serif;}
        ''')
        self.right_view_result.setStyleSheet('''
            QListView#right_view{
                font-size:18px;
                font-weight:400;
                font-family:"Helvetica Neue", Helvetica, Arial, sans-serif;}
        ''')
        
        self.setWindowOpacity(0.95) # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground) # 设置窗口背景透明
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint) # 隐藏边框
        self.main_layout.setSpacing(0)
        
        self.left_close.clicked.connect(self.close_)
        self.left_mini.clicked.connect(self.mini_)
        self.left_view.clicked.connect(self.detail_recommand)
        self.right_bar_search_input.returnPressed.connect(self.result)
        self.right_view_result.clicked.connect(self.detail_result)
        
        self.get_recommand()
        
    def get_recommand(self):
        recommand_model = QStringListModel()
        recommand_info = []
        temp = []
        with open("D:/QA/wikidata-simplequestions-master/recommend_subject.txt", "r", encoding='UTF-8') as file:
            for line in file:
                line = line.strip()
                temp.append(line.split("\t")[1])
                recommand_info.append(line)
        recommand_model.setStringList(recommand_info)
        temp_model = QStringListModel(temp)
        self.left_view.setModel(temp_model)
        self.recommand_model = recommand_model
        return recommand_model
        
    def mini_(self):
        self.showMinimized()
    def close_(self):
        self.close()
    def detail_recommand(self):
        index = self.left_view.currentIndex()
        text = self.recommand_model.data(index, 0)
        text = text.split('\t')
        id = "id:\t" + text[0] + "\n"
        label = "label:\t" + text[1] + "\n"
        des = "description:\t" + text[2] + "\n"
        question = "sample question:\t" + text[3] + "\n"
        self.right_text_detail.setText(id + label + des + question)
        self.right_label_subject.setText("当前主题:\t" + text[0])
        return text[0]
    
    def detail_result(self):
        index = self.right_view_result.currentIndex()
        spo = self.result_model.data(index, 0)
        spo = spo.split("\t")[1].split()
        out = ""
        for item in spo:
            try:
                item_dict = get_entity_dict_from_api(item)
            except Exception:
                out = out + "error in finding " + item + " in wikidata\n\n"
                continue
            if item_dict["id"] != item:
                out = out + item + " in wikidata is redirected to " + item_dict["id"] + "\n"
            try:
                id = "id:\t\t" + item_dict["id"] + "\n"
                labels = "label:\t\t" + item_dict["labels"]["en"]["value"] + "\n"
                descriptions = "description:\t" + item_dict["descriptions"]["en"]["value"] + "\n"
                out = out + id + labels + descriptions + "\n"
            except KeyError:
                out = out + item + " in wikidata cannot be displayed in English\n"
                id = "id:\t\t" + item_dict["id"] + "\n"
                key = list(item_dict["labels"].keys())[0]
                labels = "label:\t\t" + item_dict["labels"][key]["value"] + "\n"
                key = list(item_dict["descriptions"].keys())[0]
                descriptions = "description:\t" + item_dict["descriptions"][key]["value"] + "\n"
                out = out + id + labels + descriptions + "\n"
                continue
        self.right_text_detail.setText(out)
        return out
        
    def result(self):
        result_model = QStringListModel()
        question = self.right_bar_search_input.text()
        subject = self.right_label_subject.text().split("\t")[1]
        result = self.demo.qa(question, subject)
        if result =="out of length":
            result_model.setStringList(["question out of max length, which is 20"])
            self.right_view_result.setModel(result_model)
            return
        if result == "no answer":
            result_model.setStringList(["no answer found in local knowledgebase"])
            self.right_view_result.setModel(result_model)
            return
        out = []
        for r in result:
            sc = r[0]
            spo = r[1]
            s = self.demo.knowledgebase_dict[spo[0]]
            p = self.demo.knowledgebase_dict[spo[1]]
            o = self.demo.knowledgebase_dict[spo[2]]
            spo = s + " " + p + " " + o
            out.append("得分: " + str(sc) + "\t" + str(spo))
        
        result_model.setStringList(out)
        self.result_model = result_model
        self.right_view_result.setModel(result_model)
        return result_model

if __name__ == "__main__":
    print("demo.py")
    run_ui()