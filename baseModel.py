import copy
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from utils.misc import ImageLoader
import json


import six.moves.cPickle as pickle
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import  ImageLoader
from utils.nn import NN


mean_file_path = "./utils/ilsvrc_2012_mean.npy"


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build()

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data):
        """ Train the model using the COCO train2014 data. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                image_files, sentences, masks = batch
                images = self.image_loader.load_images(image_files)
                feed_dict = {self.images: images,
                             self.sentences: sentences,
                             self.masks: masks}
                # _, summary, global_step = sess.run([self.opt_op,
                #                                     self.summary,
                #                                     self.global_step],
                #                                     feed_dict=feed_dict)
                _, global_step = sess.run([self.opt_op,
                                                    self.global_step],
                                                   feed_dict=feed_dict)
                if (global_step + 1) % config.save_period == 0:
                    self.save()
                #train_writer.add_summary(summary, global_step)
            train_data.reset()

        self.save()
        train_writer.close()
        print("Training complete.")

    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
        #for k in range(1):
            batch = eval_data.next_batch()
            #caption_data = self.beam_search(sess, batch, vocabulary)
            images = self.image_loader.load_images(batch)
            caption_data, scores = sess.run([self.predictions, self.probs], feed_dict={self.images: images})
            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                ## self.predictions will return the indexes of words, we need to find the corresponding word from it.
                word_idxs = caption_data[l]
                ## get_sentence will return a sentence till there is a end delimiter which is '.'
                caption = str(vocabulary.get_sentence(word_idxs))
                results.append({'image_id': int(eval_data.image_ids[idx]),
                                'caption': caption})
                #print(results)
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = mpimg.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        fp = open(config.eval_result_file, 'w')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")

    def test_original(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        captions = []
        scores = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()
            images = self.image_loader.load_images(batch)
            caption_data,scores_data = sess.run([self.predictions,self.probs],feed_dict={self.images:images})

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                ## self.predictions will return the indexes of words, we need to find the corresponding word from it.
                word_idxs = caption_data[l]
                ## get_sentence will return a sentence till there is a end delimiter which is '.'
                caption = vocabulary.get_sentence(word_idxs)
                print(caption)
                captions.append(caption)
                scores.append(scores_data[l])

                # Save the result in an image file
                image_file = batch[l]
                image_name = image_file.split(os.sep)[-1]
                image_name = os.path.splitext(image_name)[0]
                img = mpimg.imread(image_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(caption)
                plt.savefig(os.path.join(config.test_result_dir,
                                         image_name+'_result.jpg'))

        ##Save the captions to a file
        results = pd.DataFrame({'image_files':test_data.image_files,
                                'caption':captions,
                                'prob':scores})
        results.to_csv(config.test_result_file)
        print("Testing complete.")



    # TODO (JIN): write the test process for our data
    def test(self,
             sess,
             test_data_dir,
             save_result_dir,
             vocabulary,
             test_icwsm_data=False):
        """

        :param sess:
        :param test_data_dir:
        :param save_result_dir:
        :param vocabulary:
        :return:
        """
        config = self.config

        image_loader = ImageLoader(mean_file=mean_file_path)

        if not os.path.exists(config.test_result_dir):
            os.mkdir(config.test_result_dir)

        # Test Bill's data
        if test_icwsm_data:
            # Load each sub-folder and all the JPGs in sub-folder
            for path, subdirs, files in os.walk(test_data_dir):
                for name in files:
                    # Example: /afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/invasion_triplets/
                    # 4658_triplet/chesno_movement/1826/post_1826_2022-03-01T15:00:21+00:00.json
                    one_file = os.path.join(path, name)

                    triplet = one_file.split("/")[-4]
                    user = one_file.split("/")[-3]
                    post_id = one_file.split("/")[-2]
                    file_name = one_file.split("/")[-1]

                    target_save_dir = os.path.join(save_result_dir, triplet)

                    if not os.path.isdir(target_save_dir):
                        os.mkdir(target_save_dir)
                        print("Making directory: ", target_save_dir)

                    # Check whether this is an image file
                    if name.endswith('.jpg'):
                        one_image = image_loader.load_image(os.path.join(path, name))

                        # Get word indices and probs
                        caption_data, scores_data = sess.run([self.predictions,self.probs],
                                                            feed_dict={self.images:[one_image]})

                        # Find the words according to the index
                        ## get_sentence will return a sentence till there is a end delimiter which is '.'
                        final_caption = vocabulary.get_sentence(caption_data[0])

                        # Save caption into Json file
                        file_name = file_name.split(".")[0] + "_caption.json"
                        file_full_name = user + "_" + post_id + "_" + file_name
                        file_save_path = os.path.join(target_save_dir, file_full_name)

                        data = {"description": final_caption}
                        with open(file_save_path, 'w') as fp:
                            json.dump(data, fp)

                        print("File saved: ", file_save_path)

        # TODO: Generate captions for SAFE FakeNewsDataset
        else:
            print("Using Show and Tell on FakeNewsDataset.")

            all_imgs = os.listdir(test_data_dir)

            for i, one_image_name in enumerate(all_imgs):
                # Print progress
                if (i % 100 == 0) or (i == len(all_imgs)):
                    print("Number of sample processed: ", i)

                # Get captions
                try:
                    one_image = image_loader.load_image(os.path.join(test_data_dir, one_image_name))
                except:
                    print("!! Broken image: ", i)
                    continue

                caption_data, scores_data = sess.run([self.predictions, self.probs],
                                                     feed_dict={self.images: [one_image]})

                final_caption = vocabulary.get_sentence(caption_data[0])

                # Save caption into Json file
                file_name = one_image_name.split(".")[0] + "_caption.json"
                file_save_path = os.path.join(save_result_dir, file_name)

                data = {"description": final_caption}
                with open(file_save_path, 'w') as fp:
                    json.dump(data, fp)



    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path, allow_pickle=True).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("All variables present...")
        for var in tf.all_variables():
            print(var)
        with tf.variable_scope('conv1_1',reuse = True):
            kernel = tf.get_variable('conv1_1_W')

        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path,encoding='latin1')
        count = 0
        for param_name in tqdm(data_dict.keys()):
            op_name = param_name[:-2]
            print(param_name)
            #print(op_name)
            with tf.variable_scope(op_name, reuse = True):
                try:
                    var = tf.get_variable(param_name)
                    session.run(var.assign(data_dict[param_name]))
                    count += 1
                except ValueError:
                    print("No such variable")
                    pass

        print("%d tensors loaded." %count)