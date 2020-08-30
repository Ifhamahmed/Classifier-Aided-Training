# import packages
from comet_ml import Experiment

# experiment = Experiment("m487mKQwNTqFF4z7aZRX3Xv19", project_name="UNET_EndToEnd", log_env_gpu=True)

from keras.optimizers import SGD, Adam
from Models import ResNet50, UNET_Extended, UNET
from dataLoader import Dataloader
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import pandas as pd
import numpy as np
import os
import cv2
import pandas
from tqdm import tqdm
from keras import backend as K
from DummyLayer import DummyLayer


class TrainConditional(object):
    def __init__(self, image_shape, numClasses, palette, train_mask_path, channels=None, classifier_weights=None,
                 segmentor_weights=None):
        if channels is None:
            channels = [3, 3]

        self.palette = palette
        self.image_shape = image_shape
        self.numOfClasses = numClasses
        self.classifier_weights = classifier_weights
        self.segmentor_weights = segmentor_weights

        if self.classifier_weights is None:
            raise ValueError("Classifier needs to be pretrained!!!")

        # End to End Training Parameters
        self.INIT_LR = 0.00001
        self.weight = 4
        self.training_weights = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
                                          1, 1, 0, 0, 0, 0, 0])

        # UNET Shape Parameter
        self.segmentor_input_shape = (self.image_shape[0], self.image_shape[1], channels[0])
        self.seg_shape = (self.image_shape[0], self.image_shape[1], self.numOfClasses)

        # ResNet50 Shape Parameter
        self.classifier_input_shape = (self.image_shape[0], self.image_shape[1], self.numOfClasses)

        # get Prior probability
        self.p_a_ag = self._get_prior(mask_path=train_mask_path)

        # Build Global Context Classifier
        ResNet = ResNet50()
        self.classifier = ResNet.build(self.classifier_input_shape, self.numOfClasses, self.classifier_weights)
        self.classifier.summary()

        # Build Segmentor
        self.segmentor = UNET.build(self.segmentor_input_shape, self.numOfClasses, self.segmentor_weights)
        self.segmentor.summary()

        # Define Composite
        self.composite = self._define_composite(learning_rate=self.INIT_LR)

        # Compile Segmentor
        self.segmentor_LR = 0.001
        opt = Adam(lr=self.segmentor_LR)
        self.segmentor.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    def _custom_loss(self, gt_seg, syn_seg):
        p_a_ag = self.p_a_ag
        syn_seg /= tf.reduce_sum(syn_seg,
                                 reduction_indices=len(syn_seg.get_shape()) - 1,
                                 keep_dims=True)
        weight = self.weight

        def seg_loss(y_true, y_pred):
            def _to_tensor(x, dtype):
                """Convert the input `x` to a tensor of type `dtype`.
                # Arguments
                    x: An object to be converted (numpy array, list, tensors).
                    dtype: The destination type.
                # Returns
                    A tensor.
                """
                x = tf.convert_to_tensor(x)
                if x.dtype != dtype:
                    x = tf.cast(x, dtype)
                return x

            p_a_ag_tensor = _to_tensor(p_a_ag, 'float32')
            _EPSILON = K.epsilon()
            epsilon = _to_tensor(_EPSILON, syn_seg.dtype.base_dtype)
            clipped_syn_seg = tf.clip_by_value(syn_seg, epsilon, 1. - epsilon)
            shape = tf.shape(clipped_syn_seg)

            # Reshape prior tensor
            p_a_ag_reshaped = tf.reshape(p_a_ag_tensor, (1, shape[1], shape[2], shape[3]))
            p_a_ag_tiled = tf.tile(p_a_ag_reshaped, (shape[0], 1, 1, 1))

            # Compute Cross Entropy
            cross_entropy = gt_seg * tf.log(clipped_syn_seg)

            # Create Likelihood tensor with same shape as in 'shape' variable
            tmp = tf.reshape(y_pred, (shape[0], 1, 1, shape[3]))
            likelihood = tf.tile(tmp, (1, shape[1], shape[2], 1))

            # Compute Posterior
            posterior = tf.multiply(likelihood, p_a_ag_tiled)

            # Clip Posterior
            clipped_posterior = tf.clip_by_value(posterior, epsilon, 1. - epsilon)

            # Compute Potential
            potential = weight * tf.log(clipped_posterior)

            # Apply potential to designated nodes
            potential_nodes = tf.multiply(gt_seg, potential)

            # compute loss
            loss = tf.add(cross_entropy, potential_nodes)

            return - tf.reduce_sum(loss, reduction_indices=len(clipped_syn_seg.get_shape()) - 1)

        return seg_loss

    def _custom_accuracy(self, y_true, y_pred):
        threshold = 0.4
        if threshold != 0.5:
            threshold = K.cast(threshold, y_pred.dtype)
            y_pred = K.cast(y_pred > threshold, y_pred.dtype)
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

    def _define_composite(self, learning_rate):
        # Define RGB Input
        input1 = Input(self.segmentor_input_shape)

        # Define Segmentation Maps input
        input2 = Input(self.seg_shape)

        # Freeze Classifier Weights
        self.classifier.trainable = False

        # Get Segmentor output
        output1 = self.segmentor([input1, input2])

        # Get Classifier output
        output2 = self.classifier(output1)

        # Combine the models
        model = Model(input=[input1, input2], output=output2, name='Combined Model')

        # Compile model
        opt = Adam(lr=learning_rate)
        model.compile(loss=self._custom_loss(input2, output1), optimizer=opt, metrics=[self._custom_accuracy])
        model.summary()

        return model

    def train(self, trainset, valset, epochs, n_batch, trainset_len, valset_len):
        # initialise data generator
        traingen = trainset.data_gen(should_augment=True, batch_size=n_batch)
        valgen = valset.data_gen(should_augment=False, batch_size=n_batch)

        global loss, acc, val_loss, val_acc, step_result
        loss = []
        acc = []
        val_acc = []
        val_loss = []
        steps = 0
        prev_v_loss = 100
        steps_per_epoch = trainset_len // n_batch
        val_steps = []
        step_result = []

        print('[INFO] Commencing End to End Training of Segmentor...........')
        for epoch in range(0, epochs):
            print('Epoch : {} of {}'.format(str(epoch + 1), str(epochs)))
            for _ in tqdm(range(0, steps_per_epoch)):
                # generate batch of images, masks and ground truth labels (Class Vectors)
                imgs, masks, labels = next(traingen)

                # concatenate images and masks
                concat = [imgs, masks]

                # train the composite model end to end
                step_result = self.composite.train_on_batch(concat, labels)

                # store results
                loss.append(step_result[0])
                acc.append(step_result[1])

                # add steps for the plot
                steps += 1

                # Plot The Progress
                print("Epoch: {} [Composite loss: {}, Composite Accuracy: {}%]".format(str(epoch + 1), step_result[0],
                                                                                       step_result[1] *
                                                                                       100))

            # reset traingenerators
            # shuffle images after every epoch
            traingen = trainset.data_gen(should_augment=True, batch_size=n_batch)

            # Validate after ever epoch
            v_loss, v_acc = self._validate_segmentor(valgen, valset_len, n_batch)

            # Store Validation Results
            val_loss.append(v_loss)
            val_acc.append(v_acc)

            # Checkpoint
            prev_v_loss = self._model_checkpoint(prev_v_loss, v_loss)

            # Calculate Val Steps
            val_steps.append((steps_per_epoch * (epoch + 1)))

        # Plot Results
        self._plot_results(loss, acc, val_loss, val_acc, steps, val_steps)

        # Save weights
        print('Saving Segmentor Weights..........')
        self.segmentor.save_weights('output/UNET_weights.h5')

        print("[INFO] Saving Data...................")
        data = pd.DataFrame(data={"g_loss": loss, "g_acc": acc,
                                    "steps": np.arange(steps).tolist()})
        data.to_csv("./output/training_data.csv", sep=',', index=False)
        data = pd.DataFrame(data={"val_steps": val_steps, "val_loss": val_loss, "val_acc": val_acc})
        data.to_csv("./output/validation_data.csv", sep=',', index=False)
        print("[INFO] Process Finished..............")

    def _validate_segmentor(self, valgen, valset_len, n_batch):
        val_loss = 0
        val_acc = 0

        steps = valset_len // n_batch
        print('Validating on Validation dataset................')
        for _ in tqdm(range(0, steps)):
            imgs, masks, labels = next(valgen)

            # Validate on a batch
            step_result = self.segmentor.test_on_batch([imgs, masks], masks)

            # record results
            val_loss += step_result[0]
            val_acc += step_result[1] * 100

        # Display Results
        print("[Validation loss: {}, Validation Accuracy: {}%]".format(val_loss / steps, val_acc / steps))

        # return averaged values
        return val_loss / steps, val_acc / steps

    def _model_checkpoint(self, prev_loss, c_loss, min_delta=0.0001):
        new_val_loss = prev_loss
        if (prev_loss - c_loss) > min_delta:
            print('Validation Loss improved from {} to {}, saving checkpoint weights..............'.format(round(prev_loss, 5),
                                                                                                           round(c_loss,
                                                                                                                 5)))
            print('\nSaving Weights................')
            self.segmentor.save_weights('output/Checkpoint_Weights.h5', overwrite=True)
            new_val_loss = c_loss
        else:
            print('Validation Loss did not improve from {}'.format(round(prev_loss, 5)))
        return new_val_loss

    def _plot_results(self, loss, acc, val_loss, val_acc, steps, val_steps):
        n = np.arange(0, steps)
        # plt.style.use('ggplot')
        # plt.figure()
        #
        # plt.plot(n, loss, label="Composite Loss")
        # plt.plot(n, acc, label="Composite Accuracy")
        # plt.plot(val_steps, val_loss, label="Validation Loss")
        # plt.plot(val_steps, val_acc, label="Validation Accuracy")
        #
        # plt.title("Training/Val Loss and Accuracy (Semantic Segmentation-Adversarial)")
        # plt.xlabel("Steps #")
        # plt.ylabel("Loss/Accuracy")
        #
        # plt.legend()
        # plt.savefig('output/' + 'UNET_Conditional_plot.png')

        fig, (p1, p2) = plt.subplots(1, 2)
        fig.suptitle('Training and Validation Results')
        p1.plot(n, loss, label='Composite Loss')
        p1.plot(n, acc, label='Composite Accuracy')
        p1.legend()

        p2.plot(val_steps, val_loss, label='Validation Loss')
        p2.plot(val_steps, val_acc, label='Validation Accuracy')
        p2.set_xlabel('Steps #')
        p2.set_ylabel('Loss/Accuracy')
        p2.legend()

        fig.savefig('output/' + 'UNET_Conditional_plot.png')

    def _one_hot_encode(self, mask):
        one_hot_map = []
        for color in palette:
            class_map = (color == mask)
            class_map = np.all(class_map, axis=2) * 1
            class_map = class_map.astype('float32')
            one_hot_map.append(class_map)

        return one_hot_map

    def _get_prior(self, mask_path):
        # normalization factor
        dataset_len = len(os.listdir(mask_path))

        # Bayesian term Estimator
        # first Step -> Calculate Prior

        # Estimation of Prior
        # P(A = Ag)
        size = self.seg_shape
        ohm = np.zeros(size, dtype='float32')
        n = sorted(os.listdir(mask_path))

        print('Estimating the Prior: P(A = Ag) value...........')
        for k in tqdm(range(0, dataset_len)):
            # read mask
            mask = cv2.imread(mask_path + '/' + n[k])
            mask = cv2.resize(mask, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)

            # get one hot encoded mask
            ohm_mask = self._one_hot_encode(mask)

            for i in range(0, len(ohm_mask)):
                ohm[:, :, i] += ohm_mask[i]

        p_a_ag = ohm / dataset_len
        print(p_a_ag.shape)
        print('Prior Estimation Complete')
        return p_a_ag

    def predict(self, image):
        image = cv2.imread(image)
        image = cv2.resize(image, (256, 256))
        image = image.astype("float32") / 255
        # add batch dimension
        image = image.reshape((1, image.shape[0], image.shape[1], 3))

        # predict
        preds = self.segmentor.predict(image)

        # collapse class probabilities to label map
        preds = preds.reshape((preds.shape[1], preds.shape[2], preds.shape[3]))
        preds = preds.argmax(axis=-1)

        label_map = np.zeros((preds.shape[0], preds.shape[1], 3)).astype('float32')

        for ind in range(0, len(palette)):
            submat = np.where(preds == ind)

            np.put(label_map[:, :, 0], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][0])
            np.put(label_map[:, :, 1], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][1])
            np.put(label_map[:, :, 2], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][2])

        label_map = label_map / 255
        cv2.imshow('mask', label_map)
        cv2.waitKey(0)


if __name__ == "__main__":
    # import colour palette
    df = pd.read_csv('classes.csv', ",", header=None)
    palette = np.array(df.values, dtype=np.uint8)
    num_of_Classes = palette.shape[0]

    # make output folder
    if 'output' not in os.listdir(os.getcwd()):
        os.makedirs(os.getcwd() + '/output')

    # Dataset path
    dataset_path = '/home/ifham_fyp/PycharmProjects/dataset'

    train_frame_path = os.path.sep.join([dataset_path, "train_frames/train"])
    train_mask_path = os.path.sep.join([dataset_path, "train_masks/train"])
    train_vector_path = os.path.sep.join([dataset_path, "train_vectors/train"])

    val_frame_path = os.path.sep.join([dataset_path, "val_frames/val"])
    val_mask_path = os.path.sep.join([dataset_path, "val_masks/val"])
    val_vector_path = os.path.sep.join([dataset_path, "val_vectors/val"])

    test_frame_path = os.path.sep.join([dataset_path, "test_frames/test"])
    test_mask_path = os.path.sep.join([dataset_path, "test_masks/test"])
    test_vector_path = os.path.sep.join([dataset_path, "test_vectors/test"])

    # initialise variables
    No_of_train_images = len(os.listdir(dataset_path + '/train_frames/train'))
    No_of_val_images = len(os.listdir(dataset_path + '/val_frames/val'))
    print("Number of Training Images = {}".format(No_of_train_images))
    print("Number of Validation Images = {}".format(No_of_val_images))

    # Input Size
    input_size = [512, 512]

    # Training Parameters
    BS = 2
    EPOCHS = 40

    # instantiate datagenerator class
    train_set = Dataloader(image_paths=train_frame_path,
                           mask_paths=train_mask_path,
                           vector_paths=train_vector_path,
                           image_size=input_size,
                           numclasses=num_of_Classes,
                           channels=[3, 3],
                           palette=palette,
                           seed=47)

    val_set = Dataloader(image_paths=val_frame_path,
                         mask_paths=val_mask_path,
                         vector_paths=val_vector_path,
                         image_size=input_size,
                         numclasses=num_of_Classes,
                         channels=[3, 3],
                         palette=palette,
                         seed=47)

    test_set = Dataloader(image_paths=test_frame_path,
                          mask_paths=test_mask_path,
                          vector_paths=test_vector_path,
                          image_size=input_size,
                          numclasses=num_of_Classes,
                          channels=[3, 3],
                          palette=palette,
                          seed=47)

    # Instantiate Class
    UNET_EndToEnd = TrainConditional(image_shape=(512, 512),
                                     numClasses=num_of_Classes,
                                     palette=palette,
                                     channels=[3, 3],
                                     classifier_weights='ResNet50_CS.h5',
                                     segmentor_weights='UNET_CS_weights.h5',
                                     train_mask_path=train_mask_path)

    # Conduct End to End Training
    UNET_EndToEnd.train(trainset=train_set, valset=val_set, epochs=EPOCHS, n_batch=BS,
                        trainset_len=No_of_train_images, valset_len=No_of_val_images)