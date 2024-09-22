import sys
import os
import numpy as np


class OneHotDial:
    def __init__(self, cell_n):
        self.rng = np.random.default_rng()
        self.cell_n = cell_n
        # self.dial = np.zeros(self.cell_n)
        self.decay = self.rng.random(self.cell_n)
        self.transition = np.zeros((self.cell_n, self.cell_n))
        self.reverse = np.zeros((self.cell_n, self.cell_n))

    def tic(self):
        prev_i = np.argmax(self.decay)
        self.decay = self.decay / 2
        self.transition = self.transition / 2
        i = self.make_afar_hot()
        self.transition[prev_i, i] = 1.0
        self.reverse[i, prev_i] = 1.0
        return i

    def make_afar_hot(self):
        i = np.argmin(self.decay)
        self.make_one_hot(i)
        return i

    def make_one_hot(self, i):
        # self.dial = np.zeros(self.cell_n)
        # self.dial[i] = 1
        self.decay[i] = 1.0

    def make_next_hot(self, i):
        j = np.argmax(self.transition[i])
        self.make_one_hot(j)
        return j

    def get_previous(self, i):
        if self.reverse[i].max() > 0.0:
            return np.argmax(self.reverse[i])
        else:
            return -1

    def clear_reverse(self, i):
        self.reverse[i] = np.zeros(self.cell_n)


class SequenceMemory:
    def __init__(self, ohd, feature_dim):
        self.ohd = ohd
        self.memory = {}
        self.feature_dim = feature_dim
        self.dejavu = np.zeros((feature_dim, self.ohd.cell_n))
        self.current = -1

    def memorize_features(self, features, h):
        self.memory[h] = features
        self.dejavu[:, h:h+1] = self.dejavu[:, h:h+1] + features.reshape(-1, 1)

    def retrieve_features(self, n):
        if n in self.memory:
            return self.memory[n]
        else:
            return np.zeros(self.feature_dim)

    def tic(self):
        return self.ohd.tic()

    def get_afar(self):
        return self.ohd.make_afar_hot()

    def get_next(self, i):
        h = self.ohd.make_next_hot(i)
        features = self.retrieve_features(h)
        return h, features

    def predictions(self, i):
        tmp = self.ohd.transition[i].copy()
        predictions = []
        i = 0
        while tmp.max() > 0.0:
            m = np.argmax(tmp)
            predictions.append((m, self.retrieve_features(m)))
            tmp[m] = 0.0
            i += 1
        return predictions

    def remember(self, features):
        cells = features @ self.dejavu
        if cells.max() > 0.0:
            h = np.argmax(cells)
            self.ohd.make_one_hot(h)
        else:
            h = self.ohd.make_afar_hot()
            self.memorize_features(features, h)
        return h

    def state_candidates(self, features):
        candidates_via_transition = self.ohd.transition @ self.ohd.decay
        candidates_via_match = features @ self.dejavu
        return candidates_via_transition * candidates_via_match

    def erase_traces(self, i):
        if i in self.memory:
            del self.memory[i]
        j = np.argmax(self.ohd.transition[i])
        self.ohd.transition[i, j] = 0.0
        h = self.ohd.get_previous(i)
        while h >= 0:
            if h in self.memory:
                del self.memory[h]
            self.ohd.transition[h, i] = 0.0
            self.ohd.reverse[i, h] = 0.0
            i = h
            h = self.ohd.get_previous(i)

def main():
    from PIL import Image
    args = sys.argv
    mode = args[1]  # 1: OneHotDial, 2: SequenceMemory
    cell_n = 7
    ohd = OneHotDial(cell_n)
    if mode == 1:
        j = ohd.make_afar_hot()  # init
        print(j)
        for i in range(cell_n):
            j = ohd.tic()
            print(j)
        print("sequence recall")
        j = ohd.make_afar_hot()
        print(j)
        for i in range(cell_n):
            j = ohd.make_next_hot(j)
            print(j)
    else:
        sm = SequenceMemory(ohd)
        in_folder = args[2]
        file_names = os.listdir(in_folder + '/')
        # image memorizing loop
        cnt = 0
        for file in file_names:
            if file.endswith('.jpg'):
                print(file)
                img = Image.open(in_folder + '/' + file)
                size = img.size
                base_width = size[0] * len([a for a in file_names if a.endswith('.jpg')])
                base_height = size[1] * 2
                if cnt == 0:
                    base = Image.new(img.mode, (base_width, base_height))
                base.paste(img, (cnt * size[0], 0))
                cnt = cnt + 1
                h = sm.tic()
                sm.memorize_features(np.array(img), h)
        # image retrieval loop
        h = sm.get_afar()
        array = sm.retrieve_features(h)
        base.paste(Image.fromarray(array), (0, size[1]))
        for i in range(cell_n):
            h, array = sm.get_next(h)
            base.paste(Image.fromarray(array), ((i + 1) * size[0], size[1]))
        base.save(args[3])


if __name__ == '__main__':
    main()
