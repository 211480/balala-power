import tensorflow as tf
import numpy as np
import os

def pic_read():
    '''
    狗图片读取案例
    :return: 
    '''
    #1，构造文件名队列
    file_queue = tf.train.string_input_producer(file_list)
    #2，读取与解码
    #读取
    reader = tf.WholeFileReader()
    #key-文件名, value-一张图片的原始编码
    key,value = reader.read(file_queue)
    #解码
    image = tf.image.decode_jpeg(value)
    print('key:\n', key)
    print('value:\n', value)
    print('image:\n', image)

    #批处理之前需要将图片的shape定下来，即将shape(?,?,?)中？确定
    #确定前两维—图像形状类型修改shape(?,?,?)—shape(200,200,?)
    image_resized = tf.image.resize_images(image,[200,200])
    print('image_resized:\n', image_resized)
    #确定第三维—静态形状修改shape(200,200,?)—shape(200,200,3)
    image_resized.set_shape(shape=[200,200,3])
    print('image_resized:\n',image_resized)

    #3，批处理(shape(200,200,3)—shape=(100, 200, 200, 3))
    image_batch = tf.train.batch([image_resized],batch_size=100,num_threads=1,capacity=100)
    print('image_batch:\n',image_batch)

    with tf.Session() as sess:
        # 使用了队列-开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        key_new,value_new,image_new,image_resized_new,image_batch_new = sess.run([key,value,image,image_resized,image_batch])
        print('key_new:\n', key_new)
        print('value_new:\n', value_new)
        print('image_new:\n', image_new)
        print('image_resized_new:\n', image_resized_new)
        print('image_batch_new:\n', image_batch_new)
        # 关闭线程
        coord.request_stop()
        coord.join(threads)

    return None



if __name__ == "__main__":
    #构建 路径+文件名 队列
    filename = os.listdir('H:/deep_learning/深度学习day2资料/02-代码/dog')#返回dog目录中的每一个文件名
    file_list = [os.path.join('H:/deep_learning/深度学习day2资料/02-代码/dog',file) for file in filename]#拼接路径+文件名
    print(file_list)

    pic_read()




