fp = open('results/att/att_weight_max.txt','r')
fw = open('heat_map/heat_word_max.txt','w')

k = fp.read().strip().split('\n\n')
print len(k)
sl=[];wl=[];ll=[];
for bloc in k:
#	print bloc
#	print len(bloc.strip().split('\n'))
	sent, weight, lab = bloc.strip().split('\n')
	sent = sent.split()
	weight = weight.split()
#	weight = [float(i) for i in weight]
	l = sent.index('unkown')
	sl.append(sent[:l])
	wl.append(weight[:l])
	ll.append(lab)

 
for s,w,lab in zip(sl,wl,ll):
	fw.write(' '.join(s))
	fw.write('\n')
	
	fw.write(' '.join(w))
	fw.write('\n')
	fw.write(lab)
	fw.write('\n\n')

