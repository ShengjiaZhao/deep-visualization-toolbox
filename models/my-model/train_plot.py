__author__ = 'shengjia'


def extract_info(filename):
    reader = open(filename)
    conv3_seq = []
    conv4_seq = []
    conv5_seq = []
    iteration_seq = []
    accuracy_seq = []
    while True:
        line = reader.readline().split()
        if len(line) == 0:
            break
        conv3 = float(line[2].split('(')[0])
        conv4 = float(line[3].split('(')[0])
        conv5 = float(line[4].split('(')[0])
        line = reader.readline().split()
        if len(line) == 0:
            break
        iteration = float(line[2])
        accuracy = float(line[3])

        conv3_seq.append(conv3)
        conv4_seq.append(conv4)
        conv5_seq.append(conv5)
        iteration_seq.append(iteration)
        accuracy_seq.append(accuracy)

    return {'conv3': conv3_seq,
            'conv4': conv4_seq,
            'conv5': conv5_seq,
            'iter': iteration_seq,
            'accuracy': accuracy_seq}


def running_avg(seq):
    sum = 0.0
    for i in range(len(seq)):
        sum = sum * 0.9 + seq[i]
        seq[i] = sum
