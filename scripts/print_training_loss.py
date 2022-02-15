#!/usr/bin/python
import re

def read_logs(train_log, progression_log):
    updates2dev = []
    with open(progression_log) as pfp:
        headers = None
        for line in pfp:
            if headers is None:
                headers = line.strip().split(',')
                continue
            updates,dev,train = line.strip().split(',')
            updates = int(updates)
            updates2dev.append((updates,dev))
            
    with open(train_log) as tfp:
        contents = tfp.read()
        
    last = None
    stop = False
    i = 0
    to_add = 0
    for line in contents.split('['):
        if 'Running Loss' in line:
            match = re.match('.+?Running Loss\: +([\d\.]+)\:.+?(\d+)/\d+', line.replace('\n', ''))
            loss = match.group(1)
            updates = (int(int(match.group(2))/100) * 100)
            
#            print('*last = ', last, 'updates = ', updates, 'updates+to_add = ', updates + to_add)
            
                

            # add previous # updates if necessary
            if last is not None and updates + to_add < last:
#                print('adding ' + str(last) + ' to ' + str(updates))
#                print('last = ', last)
#                input()
                to_add = last - updates# - to_add
                stop = True

            updates += to_add

            # print out dev if #updates lower than training updates
            while i < len(updates2dev):
                if updates2dev[i][0] <  updates:# + to_add:
                    print(str(updates2dev[i][0]) + '\t' + updates2dev[i][1] + '\t\t')
                    i += 1
                else:
                    break
            
            # print out if different from previous
            if last is None or last != updates:
                # print out dev score
                if i < len(updates2dev) - 1 and updates2dev[i][0] == updates:
                    print(str(updates) + '\t' + updates2dev[i][1] + '\t' + loss)
                    i += 1
                else:
                    print(str(updates) + '\t\t' + loss)
#            if stop:
#                input()
#                stop = False
            last = updates
    for up in updates2dev[i:]:
        print(str(up[0]) + '\t' + up[1] + '\t')
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_log')
    parser.add_argument('progression_log')
    args = parser.parse_args()

    read_logs(args.train_log, args.progression_log)
