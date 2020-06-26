from threading import Thread
from tkinter import *
import hashlib
import time
import cv2 as cv
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import threading
import math
import colorsys
from timeit import default_timer as timer
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from predict import CardPredictor
from multiprocessing import Process

LOG_LINE_NUM = 0

class MY_GUI():
    def __init__(self, init_window_name):
        self.init_window_name = init_window_name
        self.net = YOLO()



        self.init_window_name.title("交通监测系统_v1.2")           #窗口名
        #self.init_window_name.geometry('320x160+10+10')                         #290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        self.init_window_name.geometry('1500x770+10+10')
        #self.init_window_name["bg"] = "pink"                                    #窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        #self.init_window_name.attributes("-alpha",0.9)                          #虚化，值越小虚化程度越高
        #标签
        self.init_data_label = Label(self.init_window_name, text="分析过程")
        self.init_data_label.grid(row=0, column=0)
        self.result_data_label = Label(self.init_window_name, text="输出视频")
        self.result_data_label.grid(row=0, column=12)
        self.log_label = Label(self.init_window_name, text="解析结果")
        self.log_label.grid(row=12, column=0)
        self.img_info = Text(self.init_window_name)
        self.img_info.grid(row=68, column=0)

        #文本框
        # self.init_data_Text = Text(self.init_window_name, width=67, height=25)  #原始数据录入框
        # self.init_data_Text.grid(row=1, column=0, rowspan=10, columnspan=10)
        # self.result_data_Text = Text(self.init_window_name, width=70, height=49)  #处理结果展示
        # self.result_data_Text.grid(row=1, column=12, rowspan=15, columnspan=10)
        self.log_data = Text(self.init_window_name, width=70, height=25)  # 日志框
        self.log_data.grid(row=13, column=0, columnspan=10)
        # self.frame= Frame(self.init_window_name, width=66, height=9)  # 日志框
        # self.frame.grid(row=10, column=2, columnspan=10)
        self.canvas1 = Canvas(self.init_window_name, bg='white', width=1000, height=800)
        self.canvas1.grid(row=1, column=12, rowspan=15, columnspan=10)
        self.canvas2 = Canvas(self.init_window_name, bg='white', width=500, height=300)
        self.canvas2.grid(row=1, column=0, rowspan=10, columnspan=10)
        #  canvas1.place(x=imagepos_x, y=imagepos_y)
        #  按钮
        self.str_trans_to_md5_button = Button(self.init_window_name, text="解析", bg="lightblue", width=10,
                                              command=lambda : self.thread_it1(self.str_trans_to_md5()))  # 调用内部方法  加()为直接调用
        self.str_trans_to_md5_button.grid(row=4, column=11)
        self.str_trans_to_md5_button = Button(self.init_window_name, text="打开视频", bg="lightblue", width=10,
                                              command=self.from_pic)  # 调用内部方法  加()为直接调用
        self.str_trans_to_md5_button.grid(row=2, column=11)
        self.str_trans_to_md5_button = Button(self.init_window_name, text="播放分析视频", bg="lightblue", width=10,
                                              command=self.from_pic1)  # 调用内部方法  加()为直接调用
        self.str_trans_to_md5_button.grid(row=3, column=11)

    #设置窗口
    def from_pic(self):
        self.threadRun = False
        self.video_path = askopenfilename(title="选择识别路径",filetypes=[("avi","*.avi"),("mp4","*.mp4"),("png","*.png"),("jpg","*.jpg"),])

    def tkImage(self, vc):
        ref, frame = vc.read()
        image_width, image_height = frame.shape[1], frame.shape[0]
        cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(cvimage)
        pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
        tkImage = ImageTk.PhotoImage(image=pilImage)
        return tkImage

    #  获取当前时间
    def get_current_time(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        return current_time
    #  日志动态打印
    def write_log_to_Text(self, logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) + '  ' + str(logmsg) + '\n'
        if LOG_LINE_NUM <= 15:
            self.log_data.insert(END, logmsg_in)
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.log_data.delete(1.0,2.0)
            self.log_data.insert(END, logmsg_in)

    def set_test(self, img, img_info):

        # a, b =img.shape()
        # print(a,b)
        # height, width = img.shape[:2]
        # #
        # # # 缩小图像
        # size = (int(width * 0.3), int(height * 0.5))

        self.write_log_to_Text(img_info)
        pilImage = Image.fromarray(img)

        #print(type(pilImage))
        #print(pilImage.size)
        # height, width = img.shape[:2]
        # pilImage.size[1]=(int(width * 0.3), int(height * 0.5))
        # h=pilImage.size[0]
        # w=pilImage.size[1]
        pilImage = pilImage.resize((500,300),Image.ANTIALIAS)
        tkImage= ImageTk.PhotoImage(image=pilImage)
#       tkImage=cv2.resize(tkImage,(tkImage.height()*0.5,tkImage.width()*0.5),Image.ANTIALIAS)
        self.canvas2.create_image(0, 0, anchor='nw', image=tkImage)
        self.init_window_name.update_idletasks()  # 最重要的更新是靠这两句来实现
        self.init_window_name.update()
        #label["text"] = "My name is Ben"
        #picture1 = self.tkImage(cap)
        #self.canvas1.create_image(0, 0, anchor='nw', image=picture1)


    def from_pic1(self):
        cap = cv.VideoCapture('video-01.avi')
        def video_loop():
            try:
                while True:
                    picture1 = self.tkImage(cap)
                    self.canvas1.create_image(0, 0, anchor='nw', image=picture1)
                    self.init_window_name.update_idletasks()  # 最重要的更新是靠这两句来实现
                    self.init_window_name.update()
            except:
                pass

        video_loop()
        self.init_window_name.mainloop()
        cap.release()
        # cv2.destroyAllWindows()
        while cap.isOpened():
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # show gray picture
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # cv.imshow('frame', gray)
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    def thread_it1(self, func, *args):  # 传入函数名和参数
        # 创建线程
        t = Thread(target=func, args=args)
        # 守护线程
        t.setDaemon(True)
        # 启动
        t.start()
    # class MyThread(threading.Thread):
    #     def __init__(self, func, *args):
    #         super().__init__()
    #         self.func = func
    #         self.args = args
    #         self.setDaemon(True)
    #         self.start()    # 在这里开始
    #     def run(self):
    #         self.func(*self.args)

    #功能函数
    def str_trans_to_md5(self):

        # t11 = Process(target=self.detect_video(self.video_path, 'fortest.avi'), args=('nick',))
        # t11.start()
        self.detect_video(self.video_path, 'fortest.avi')

        src = self.init_data_Text.get(1.0,END).strip().replace("\n","").encode()
        if src:
            try:
                myMd5 = hashlib.md5()
                myMd5.update(src)
                myMd5_Digest = myMd5.hexdigest()
                #print(myMd5_Digest)
                #输出到界面
                self.result_data_Text.delete(1.0,END)
                self.result_data_Text.insert(1.0,myMd5_Digest)
                self.write_log_to_Text("INFO:解析成功")
            except:
                self.result_data_Text.delete(1.0,END)
                self.result_data_Text.insert(1.0,"解析")
        else:
            self.write_log_to_Text("ERROR:解析失败")

    def detect_video(self, video_path, output_path=""):
        #writer = open('temp.txt','a+')
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            try:
                image = Image.fromarray(frame)
            except Exception:
                print('finish')
                break
            maige, result_info = self.net.detect_image(image, frame)
            #t = Thread(target=self.set_test(result_info), args=('nick',))
            #t.start()

            #writer.write(result_info)
            #writer.write('\n')

            result = np.asarray(image)
            self.set_test(result, result_info)   #!!!!!!!!

            # curr_time = timer()
            # exec_time = curr_time - prev_time
            # prev_time = curr_time
            # accum_time = accum_time + exec_time
            # curr_fps = curr_fps + 1
            # if accum_time > 1:
            #    accum_time = accum_time - 1
            #    fps = "FPS: " + str(curr_fps)
            #    curr_fps = 0
            #cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #            fontScale=0.50, color=(255, 0, 0), thickness=2)
            #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            #cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # self.net.close_session()

        # writer.close()


class YOLO():
    _defaults = {
        "model_path": 'model_data/yolo.h5', #yolo
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt', #myclasses
        "score" : 0.3,
        "iou" : 0.2,  # 0.45
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.predictor = CardPredictor()
        self.points = []

    def detect_carnumber(self,img_bgr, box):  #r1, r2, r3, r4
        img_bgr = img_bgr[box[0]:box[1], box[2]:box[3]]
        r = self.predictor.predict(img_bgr)
        return r

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, frame):
        start = timer()
        #origin_img = image
        #origin_img = cv2.cvtColor(np.asarray(origin_img), cv2.COLOR_RGB2BGR)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        #out_boxes = [i for i in out_boxes if i in ['bus','car','person']]
        #out_scores = [i for i in out_scores if i in ['bus','car','person']]
        out_classes = [i for i in out_classes if self.class_names[i] in ['bus','car','person']]
        result_info = 'Found {} boxes for {}'.format(len(out_boxes), 'img')
        print(result_info)
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 400

        if len(self.points) != len(out_classes):
            for i, c in reversed(list(enumerate(out_classes))):
                box = out_boxes[i]
                top, left, bottom, right = box

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                self.points.append({'class': self.class_names[c], 'point': [(top+bottom)/2, (left+right)/2]})

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if predicted_class == 'car' or predicted_class == 'bus':
                box = [top, bottom, left, right]
                carpoint = [(top+bottom)/2, (left+right)/2]
                rate = 51
                while rate>50:
                    if self.points[i]['class'] == predicted_class:
                        rate = math.sqrt((carpoint[0]-self.points[i]['point'][0])**2 + (carpoint[1]-self.points[i]['point'][1])**2)/2
                    i += 1
                    if i == len(self.points):
                        rate = 0
                        break

                carnumber = self.detect_carnumber(frame, box)

                if carnumber is None:
                    carnumber = 'no'
                else:
                    carnumber = ''.join(carnumber)
                label = '{} {} {:.2f}'.format(predicted_class, carnumber, rate)  #label = '{} {:.2f}'
            else:
                label = '{} {:.1f} {:.2f}'.format(predicted_class, score, rate)  #label = '{} {:.2f}'
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            print(label)#(left, top), (right, bottom)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw


        end = timer()
        print(end - start)
        return image, result_info

    def close_session(self):
        self.sess.close()


if __name__ == "__main__":
    init_window = Tk()  # 实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    #ZMJ_PORTAL.set_init_window()
    init_window.mainloop()  # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示