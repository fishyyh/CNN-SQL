f=open("train_sql.txt1",'a',encoding="utf-8")
for lines in open("data/train_sql.txt",'r',encoding="utf-8"):
	if len(lines):
		f.write(lines)
