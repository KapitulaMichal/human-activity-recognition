import cv2
import numpy as np
import math

class KTH:
    NUMBER_OF_FRAMES = 8
    FRAME_INCREMENT = 2

    index = 0

    train_images = None
    train_labels = None

    test_images = None
    test_labels = None

    def create_new_dataset(self):
        sequences = []

        labels_dict = {
            'boxing': np.array([1, 0, 0, 0, 0, 0]),
            'handcl': np.array([0, 1, 0, 0, 0, 0]),
            'handwa': np.array([0, 0, 1, 0, 0, 0]),
            'joggin': np.array([0, 0, 0, 1, 0, 0]),
            'runnin': np.array([0, 0, 0, 0, 1, 0]),
            'walkin': np.array([0, 0, 0, 0, 0, 1])
        }

        with open('sequences.txt') as fp:
            for line in fp:
                if not line == '\n':
                    split = line.split()
                    split.remove('frames')
                    for index, value in enumerate(split):
                        if value[-1] == ',':
                            split[index] = split[index][:-1]

                    sequences.append(split)

        for current_sequence in sequences:

            if not self.train_images is None:
                print(self.train_images.shape, self.train_labels.shape)

            if not self.test_images is None:
                print(self.test_images.shape, self.test_labels.shape)

            file = current_sequence[0]
            print(file)
            frame_sets = current_sequence[1:]

            proper_label = labels_dict[file[9:15]]

            cap = cv2.VideoCapture('./all_actions/' + file + '_uncomp.avi')

            frame_count = 1

            ret, frame1 = cap.read()

            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255

            for frame_set in frame_sets:
                start = int(frame_set.split('-')[0])
                stop = int(frame_set.split('-')[1])

                if frame_count < start:

                    while frame_count < start:
                        ret, frame2 = cap.read()
                        current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                        prvs = current_frame

                        frame_count += 1

                while True:

                    if frame_count + self.NUMBER_OF_FRAMES * self.FRAME_INCREMENT <= stop:

                        frames_to_stack = []
                        optical_flow_to_stack = []
                        Dog_to_stack = []

                        for frame in range(self.NUMBER_OF_FRAMES):
                            #retrieve frame
                            ret, frame2 = cap.read()
                            current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                            #optical flow
                            flow = cv2.calcOpticalFlowFarneback(prvs, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                            imgScale = 0.25
                            # bgr = cv2.resize(bgr, (int(bgr.shape[1] * imgScale), int(bgr.shape[0] * imgScale)))
                            bgr = cv2.resize(bgr, (80, 60))

                            prvs = current_frame

                            # Difference of Gaussian
                            # current_frame = cv2.resize(current_frame, (int(current_frame.shape[1] * imgScale), int(current_frame.shape[0] * imgScale)))
                            current_frame = cv2.resize(current_frame, (80,60))

                            blur9 = cv2.GaussianBlur(current_frame, (9, 9), 0)
                            blur7 = cv2.GaussianBlur(current_frame, (7, 7), 0)

                            DoGim = blur9 - blur7

                            frames_to_stack.append(current_frame)
                            optical_flow_to_stack.append(bgr)
                            Dog_to_stack.append(DoGim)

                            frame_count += 1

                            if self.FRAME_INCREMENT > 1:
                                for _ in range(self.FRAME_INCREMENT - 1):
                                    ret, frame2 = cap.read()
                                    current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                                    prvs = current_frame

                                    frame_count += 1

                        frames_stack = np.stack(frames_to_stack, axis=2)
                        optical_flow_stack = np.stack(optical_flow_to_stack, axis=2)
                        Dog_stack = np.stack(Dog_to_stack, axis=2)

                        full_example = np.stack([frames_stack, optical_flow_stack, Dog_stack], axis=3)

                        if int(file[6:8]) <= 20:
                            if self.train_images is None:
                                self.train_images = np.array([full_example])
                                self.train_labels = np.array([proper_label])
                            else:
                                self.train_images = np.append(self.train_images, [full_example], axis=0)
                                self.train_labels = np.append(self.train_labels, [proper_label], axis=0)


                        else:
                            if self.test_images is None:
                                self.test_images = np.array([full_example])
                                self.test_labels = np.array([proper_label])
                            else:
                                self.test_images = np.append(self.test_images, [full_example], axis=0)
                                self.test_labels = np.append(self.test_labels, [proper_label], axis=0)

                    else:
                        break


        #self.train_images = tuple(self.train_images)
        #self.test_images = tuple(self.test_images)

        print(self.train_labels.shape)

        np.save('train_images_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_inc_' + str(self.FRAME_INCREMENT) + '.npy', self.train_images)
        np.save('train_labels_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_inc_' + str(self.FRAME_INCREMENT) + '.npy', self.train_labels)

        np.save('test_images_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_inc_' + str(self.FRAME_INCREMENT) + '.npy', self.test_images)
        np.save('test_labels_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_inc_' + str(self.FRAME_INCREMENT) + '.npy', self.test_labels)

        print(self.train_images.shape, self.train_labels.shape)
        print(self.test_images.shape, self.test_labels.shape)

    def create_new_dataset_single(self):
        sequences = []

        labels_dict = {
            'boxing': np.array([1, 0, 0, 0, 0, 0]),
            'handcl': np.array([0, 1, 0, 0, 0, 0]),
            'handwa': np.array([0, 0, 1, 0, 0, 0]),
            'joggin': np.array([0, 0, 0, 1, 0, 0]),
            'runnin': np.array([0, 0, 0, 0, 1, 0]),
            'walkin': np.array([0, 0, 0, 0, 0, 1])
        }

        with open('sequences.txt') as fp:
            for line in fp:
                if not line == '\n':
                    split = line.split()
                    split.remove('frames')
                    for index, value in enumerate(split):
                        if value[-1] == ',':
                            split[index] = split[index][:-1]

                    sequences.append(split)

        for current_sequence in sequences:

            if not self.train_images is None:
                print(self.train_images.shape, self.train_labels.shape)

            if not self.test_images is None:
                print(self.test_images.shape, self.test_labels.shape)

            file = current_sequence[0]
            print(file)
            frame_sets = ['1-20']
            # print(frame_sets)

            proper_label = labels_dict[file[9:15]]

            cap = cv2.VideoCapture('./all_actions/' + file + '_uncomp.avi')

            frame_count = 1

            ret, frame1 = cap.read()

            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255

            for frame_set in frame_sets:
                start = int(frame_set.split('-')[0])
                stop = int(frame_set.split('-')[1])

                if frame_count < start:

                    while frame_count < start:
                        ret, frame2 = cap.read()
                        current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                        prvs = current_frame

                        frame_count += 1

                while True:

                    if frame_count + self.NUMBER_OF_FRAMES * self.FRAME_INCREMENT <= stop:

                        frames_to_stack = []
                        optical_flow_to_stack = []
                        Dog_to_stack = []

                        for frame in range(self.NUMBER_OF_FRAMES):
                            #retrieve frame
                            ret, frame2 = cap.read()
                            current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                            #optical flow
                            flow = cv2.calcOpticalFlowFarneback(prvs, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                            imgScale = 0.25
                            # bgr = cv2.resize(bgr, (int(bgr.shape[1] * imgScale), int(bgr.shape[0] * imgScale)))
                            bgr = cv2.resize(bgr, (80, 60))

                            prvs = current_frame

                            # Difference of Gaussian
                            # current_frame = cv2.resize(current_frame, (int(current_frame.shape[1] * imgScale), int(current_frame.shape[0] * imgScale)))
                            current_frame = cv2.resize(current_frame, (80,60))

                            blur9 = cv2.GaussianBlur(current_frame, (9, 9), 0)
                            blur7 = cv2.GaussianBlur(current_frame, (7, 7), 0)

                            DoGim = blur9 - blur7

                            frames_to_stack.append(current_frame)
                            optical_flow_to_stack.append(bgr)
                            Dog_to_stack.append(DoGim)

                            frame_count += 1

                            if self.FRAME_INCREMENT > 1:
                                for _ in range(self.FRAME_INCREMENT - 1):
                                    ret, frame2 = cap.read()
                                    current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                                    prvs = current_frame

                                    frame_count += 1

                        frames_stack = np.stack(frames_to_stack, axis=2)
                        optical_flow_stack = np.stack(optical_flow_to_stack, axis=2)
                        Dog_stack = np.stack(Dog_to_stack, axis=2)

                        full_example = np.stack([frames_stack, optical_flow_stack, Dog_stack], axis=3)

                        if int(file[6:8]) <= 20:
                            if self.train_images is None:
                                self.train_images = np.array([full_example])
                                self.train_labels = np.array([proper_label])
                            else:
                                self.train_images = np.append(self.train_images, [full_example], axis=0)
                                self.train_labels = np.append(self.train_labels, [proper_label], axis=0)


                        else:
                            if self.test_images is None:
                                self.test_images = np.array([full_example])
                                self.test_labels = np.array([proper_label])
                            else:
                                self.test_images = np.append(self.test_images, [full_example], axis=0)
                                self.test_labels = np.append(self.test_labels, [proper_label], axis=0)

                    else:
                        break


        #self.train_images = tuple(self.train_images)
        #self.test_images = tuple(self.test_images)

        print(self.train_labels.shape)

        np.save('train_images_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_inc_' + str(self.FRAME_INCREMENT) + '_single.npy', self.train_images)
        np.save('train_labels_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_inc_' + str(self.FRAME_INCREMENT) + '_single.npy', self.train_labels)

        np.save('test_images_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_inc_' + str(self.FRAME_INCREMENT) + '_single.npy', self.test_images)
        np.save('test_labels_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_inc_' + str(self.FRAME_INCREMENT) + '_single.npy', self.test_labels)

        print(self.train_images.shape, self.train_labels.shape)
        print(self.test_images.shape, self.test_labels.shape)

    def extract_specific_people(self):
        sequences = []
        person_images = None
        person_labels = None

        labels_dict = {
            'boxing': np.array([1, 0, 0, 0, 0, 0]),
            'handcl': np.array([0, 1, 0, 0, 0, 0]),
            'handwa': np.array([0, 0, 1, 0, 0, 0]),
            'joggin': np.array([0, 0, 0, 1, 0, 0]),
            'runnin': np.array([0, 0, 0, 0, 1, 0]),
            'walkin': np.array([0, 0, 0, 0, 0, 1])
        }

        with open('sequences.txt') as fp:
            for line in fp:
                if not line == '\n':
                    split = line.split()
                    split.remove('frames')
                    for index, value in enumerate(split):
                        if value[-1] == ',':
                            split[index] = split[index][:-1]

                    sequences.append(split)

        current_person = 1

        for current_sequence in sequences:

            file = current_sequence[0]
            print(file)
            frame_sets = current_sequence[1:]

            proper_label = labels_dict[file[9:15]]

            cap = cv2.VideoCapture('./all_actions/' + file + '_uncomp.avi')

            frame_count = 1

            ret, frame1 = cap.read()

            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255

            for frame_set in frame_sets:
                start = int(frame_set.split('-')[0])
                stop = int(frame_set.split('-')[1])

                while True:

                    if frame_count > start + 1:
                        ret, frame2 = cap.read()
                        current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                        prvs = current_frame

                        frame_count += 1

                    if frame_count + self.NUMBER_OF_FRAMES < stop:

                        frames_to_stack = []
                        optical_flow_to_stack = []
                        Dog_to_stack = []

                        for frame in range(self.NUMBER_OF_FRAMES):
                            #retrieve frame
                            ret, frame2 = cap.read()
                            current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                            #optical flow
                            flow = cv2.calcOpticalFlowFarneback(prvs, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                            imgScale = 0.25
                            # bgr = cv2.resize(bgr, (int(bgr.shape[1] * imgScale), int(bgr.shape[0] * imgScale)))
                            bgr = cv2.resize(bgr, (80, 60))

                            prvs = current_frame

                            # Difference of Gaussian
                            # current_frame = cv2.resize(current_frame, (int(current_frame.shape[1] * imgScale), int(current_frame.shape[0] * imgScale)))
                            current_frame = cv2.resize(current_frame, (80,60))

                            blur9 = cv2.GaussianBlur(current_frame, (9, 9), 0)
                            blur7 = cv2.GaussianBlur(current_frame, (7, 7), 0)

                            DoGim = blur9 - blur7

                            frames_to_stack.append(current_frame)
                            optical_flow_to_stack.append(bgr)
                            Dog_to_stack.append(DoGim)

                            frame_count += 1

                        frames_stack = np.stack(frames_to_stack, axis=2)
                        optical_flow_stack = np.stack(optical_flow_to_stack, axis=2)
                        Dog_stack = np.stack(Dog_to_stack, axis=2)

                        full_example = np.stack([frames_stack, optical_flow_stack, Dog_stack], axis=3)

                        '''
                        if int(file[6:8]) <= 20:
                            if self.train_images is None:
                                self.train_images = np.array([full_example])
                                self.train_labels = np.array([proper_label])
                            else:
                                self.train_images = np.append(self.train_images, [full_example], axis=0)
                                self.train_labels = np.append(self.train_labels, [proper_label], axis=0)


                        else:
                            if self.test_images is None:
                                self.test_images = np.array([full_example])
                                self.test_labels = np.array([proper_label])
                            else:
                                self.test_images = np.append(self.test_images, [full_example], axis=0)
                                self.test_labels = np.append(self.test_labels, [proper_label], axis=0)
                        '''

                        if int(file[6:8]) == current_person:
                            if person_images is None:
                                person_images = np.array([full_example])
                                person_labels = np.array([proper_label])
                            else:
                                person_images = np.append(person_images, [full_example], axis=0)
                                person_labels = np.append(person_labels, [proper_label], axis=0)

                        else:
                            if current_person < 10:
                                current_person_str = '0' + str(current_person)
                            else:
                                current_person_str = str(current_person)

                            np.save('specific_person_actions/person' + current_person_str + '_images_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(person_images.shape[1]) + 'x' + str(person_images.shape[2]) +'.npy', person_images)
                            np.save('specific_person_actions/person' + current_person_str + '_labels_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(person_images.shape[1]) + 'x' + str(person_images.shape[2]) + '.npy', person_labels)

                            person_images = None
                            person_labels = None

                            current_person += 1

                    else:
                        current_person_str = str(current_person)
                        break


        #self.train_images = tuple(self.train_images)
        #self.test_images = tuple(self.test_images)

        print(person_images.shape)

        np.save('specific_person_actions/person' + current_person_str + '_images_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(person_images.shape[1]) + 'x' + str(person_images.shape[2]) + '.npy', person_images)
        np.save('specific_person_actions/person' + current_person_str + '_labels_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(person_images.shape[1]) + 'x' + str(person_images.shape[2]) + '.npy', person_labels)

    def load_from_file(self):

        temp_images = np.load('train_images_8_frames_60x80_inc_2.npy')
        temp_labels = np.load('train_labels_8_frames_60x80_inc_2.npy')

        random_seed = np.arange(temp_images.shape[0])
        np.random.shuffle(random_seed)

        self.train_images = temp_images[random_seed]
        self.train_labels = temp_labels[random_seed]


        temp_images = np.load('test_images_8_frames_60x80_inc_2.npy')
        temp_labels = np.load('test_labels_8_frames_60x80_inc_2.npy')

        random_seed = np.arange(temp_images.shape[0])
        np.random.shuffle(random_seed)

        self.test_images = temp_images[random_seed][:1000]
        self.test_labels = temp_labels[random_seed][:1000]

        # self.test_images = temp_images[random_seed]
        # self.test_labels = temp_labels[random_seed]

    def load_specific_class(self, searched_class):

        labels_dict = {
            'boxing': np.array([1, 0, 0, 0, 0, 0]),
            'handcl': np.array([0, 1, 0, 0, 0, 0]),
            'handwa': np.array([0, 0, 1, 0, 0, 0]),
            'joggin': np.array([0, 0, 0, 1, 0, 0]),
            'runnin': np.array([0, 0, 0, 0, 1, 0]),
            'walkin': np.array([0, 0, 0, 0, 0, 1])
        }

        temp_images = np.load('train_images_8_frames_60x80_inc_2.npy')
        temp_labels = np.load('train_labels_8_frames_60x80_inc_2.npy')

        random_seed = np.arange(temp_images.shape[0])
        np.random.shuffle(random_seed)

        self.train_images = temp_images[random_seed]
        self.train_labels = temp_labels[random_seed]


        temp_images = np.load('test_images_8_frames_60x80_inc_2.npy')
        temp_labels = np.load('test_labels_8_frames_60x80_inc_2.npy')

        rows_with_with_class = []

        for i in range(temp_labels.shape[0]):
            if np.argmax(temp_labels[i]) == np.argmax(labels_dict[searched_class]):
                rows_with_with_class.append(i)

        length = len(rows_with_with_class)

        final = length - (length - math.floor(length/50) * 50)

        self.test_images = temp_images[rows_with_with_class]
        self.test_labels = temp_labels[rows_with_with_class]

        self.test_images = self.test_images[:final]
        self.test_labels = self.test_labels[:final]


    def load_specific(self, train_list, test_list):

        for i in train_list:
            # print(i)

            if i < 10:
                i_str = '0' + str(i)
            else:
                i_str = str(i)

            if self.train_images is None:
                self.train_images = np.load('specific_person_actions/person' + i_str + '_images_8_frames_60x80.npy')
                self.train_labels = np.load('specific_person_actions/person' + i_str + '_labels_8_frames_60x80.npy')

            else:
                self.train_images = np.append(self.train_images, np.load('specific_person_actions/person' + i_str + '_images_8_frames_60x80.npy'), axis=0)
                self.train_labels = np.append(self.train_labels, np.load('specific_person_actions/person' + i_str + '_labels_8_frames_60x80.npy'), axis=0)

        random_seed = np.arange(self.train_images.shape[0])
        np.random.shuffle(random_seed)

        self.train_images = self.train_images[random_seed]
        self.train_labels = self.train_labels[random_seed]

        for i in test_list:
            # print(i)

            if i < 10:
                i_str = '0' + str(i)
            else:
                i_str = str(i)

            if self.test_images is None:
                self.test_images = np.load('specific_person_actions/person' + i_str + '_images_8_frames_60x80.npy')
                self.test_labels = np.load('specific_person_actions/person' + i_str + '_labels_8_frames_60x80.npy')

            else:
                self.test_images = np.append(self.test_images, np.load('specific_person_actions/person' + i_str + '_images_8_frames_60x80.npy'), axis=0)
                self.test_labels = np.append(self.test_labels, np.load('specific_person_actions/person' + i_str + '_labels_8_frames_60x80.npy'), axis=0)


        random_seed = np.arange(self.test_images.shape[0])
        np.random.shuffle(random_seed)

        self.test_images = self.test_images[random_seed][:1000]
        self.test_labels = self.test_labels[random_seed][:1000]

    def normalize(self):

        self.train_images = self.train_images.astype('float32')
        self.test_images = self.test_images.astype('float32')

        self.train_images /= 255
        self.test_images /= 255

        # self.train_images = np.load('train_images_8_frames_60x80_3.npy')
        # self.test_images = np.load('test_images_8_frames_60x80_3.npy')
        #
        #
        #
        # for i in range(self.train_images.shape[0]):
        #     print('train_images: ', i,'/', self.train_images.shape[0], sep='')
        #     for j in range(self.train_images.shape[1]):
        #         for k in range(self.train_images.shape[2]):
        #             for l in range(self.train_images.shape[3]):
        #                 for m in range(self.train_images.shape[4]):
        #                     self.train_images[i][j][k][l][m] = ((self.train_images[i][j][k][l][m] / 255) - 0.5) * 2
        #
        # np.save('train_images_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_normalized_2.npy', self.train_images)
        #
        #
        #
        #
        # for i in range(self.test_images.shape[0]):
        #     print('test_images: ', i,'/', self.test_images.shape[0], sep='')
        #     for j in range(self.test_images.shape[1]):
        #         for k in range(self.test_images.shape[2]):
        #             for l in range(self.test_images.shape[3]):
        #                 for m in range(self.test_images.shape[4]):
        #                     self.test_images[i][j][k][l][m] = ((self.test_images[i][j][k][l][m] / 255) - 0.5) * 2
        #
        # np.save('test_images_' + str(self.NUMBER_OF_FRAMES) + '_frames_' + str(self.train_images.shape[1]) + 'x' + str(self.train_images.shape[2]) + '_normalized_2.npy', self.test_images)
        #
        #
        #
        #
        # # np.multiply(self.train_images, (1/255))
        # # np.multiply(self.train_labels, (1/255))
        # #
        # # np.multiply(self.test_images, (1/255))
        # # np.multiply(self.test_labels, (1/255))

    def drop_one_case(self):

        labels_dict = {
            'boxing': np.array([1, 0, 0, 0, 0, 0]),
            'handcl': np.array([0, 1, 0, 0, 0, 0]),
            'handwa': np.array([0, 0, 1, 0, 0, 0]),
            'joggin': np.array([0, 0, 0, 1, 0, 0]),
            'runnin': np.array([0, 0, 0, 0, 1, 0]),
            'walkin': np.array([0, 0, 0, 0, 0, 1])
        }

        rows_to_delete = []

        for i in range(0, len(self.train_images)):
            if self.train_labels[i][5] == 1:
                rows_to_delete.append(i)

        self.train_images = np.delete(self.train_images, rows_to_delete, 0)
        self.train_labels = np.delete(self.train_labels, rows_to_delete, 0)

        rows_to_delete = []

        for i in range(0, len(self.test_images)):
            if self.test_labels[i][5] == 1:
                rows_to_delete.append(i)

        self.test_images = np.delete(self.test_images, rows_to_delete, 0)
        self.test_labels = np.delete(self.test_labels, rows_to_delete, 0)

        self.test_images = self.test_images[:1000]
        self.test_labels = self.test_labels[:1000]

    def next_batch(self, size):
        if self.index + size < self.train_images.shape[0]:
            ret = (self.train_images[self.index : self.index + size], self.train_labels[self.index : self.index + size])
        else:
            self.index = 0
            ret = (self.train_images[self.index: self.index + size], self.train_labels[self.index: self.index + size])

        self.index += size

        return ret

    def next_batch_grey(self, size):
        if self.index + size < self.train_images.shape[0]:
            ret = (self.train_images[self.index : self.index + size, :,:,:,:1], self.train_labels[self.index : self.index + size])
        else:
            self.index = 0
            ret = (self.train_images[self.index: self.index + size, :,:,:,:1], self.train_labels[self.index: self.index + size])

        self.index += size

        return ret



def main():
    kth = KTH()
    # kth.create_new_dataset()
    # kth.create_new_dataset_single()
    kth.load_from_file()
    # kth.extract_specific_people()
    # kth.load_specific(list(range(1, 10)), list(range(21, 26)))

    # kth.load_specific_class('boxing')
    # kth.load_specific_class('handcl')
    # kth.load_specific_class('handwa')
    # kth.load_specific_class('joggin')
    # kth.load_specific_class('runnin')
    # kth.load_specific_class('walkin')

    # print(kth.train_images.shape)
    # print(kth.train_labels.shape, '\n')

    # print(kth.test_images.shape)
    # print(kth.test_labels.shape, '\n\n')

    # kth.drop_one_case()
    # kth.normalize()

    # print(kth.train_images.shape)
    # print(kth.train_labels.shape, '\n')

    print(kth.test_images.shape)
    print(kth.test_labels.shape)


    # print(kth.test_labels[:100])

    # print('\n\n\n', int(len(kth.test_images)/100))

    #print('\n\n')

    #print(kth.next_batch_grey(10)[0].shape)

    # print("%d bytes" % (kth.train_images.size * kth.train_images.itemsize))

    # for i in range(10):
        # print(kth.test_labels[i])

if __name__ == '__main__':
   main()