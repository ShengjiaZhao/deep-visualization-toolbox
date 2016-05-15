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
        conv3 = float(line[2].split('('))
        conv4 = float(line[3].split('('))
        conv5 = float(line[4].split('('))
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

    return [{'name': 'conv3', 'value': conv3_seq},
            {'name': 'conv4', 'value': conv4_seq},
            {'name': 'conv5', 'value': conv5_seq},
            {'name': 'iter', 'value': iteration_seq},
            {'name': 'accuracy', 'value': accuracy_seq}]


def running_avg(seq):
    sum = 0.0
    for i in range(len(seq)):
        sum = sum * 0.9 + seq[i]
        seq[i] = sum
