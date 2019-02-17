from test import init,check,convert2label
import time
model,w_model=init()
count=0
sql=0
xss=0
normal=0
start=time.time()
for line in open('data/test_normal.txt','r'):
    if len(line):
        count+=1
        checked=check(model,w_model,line)
        print(checked)
        if checked==0:
            normal+=1
        elif checked==1:
            sql+=1
        else :
            xss+=1
end=time.time()
print("normal %f"%(normal/count)," xss:",xss," sql:",sql)
print("sql %f"%(sql/count)," normal:",normal," xss:",xss)
print("xss %f"%(xss/count)," sql:",sql," normal:",normal)
print("cost time is "+str(end-start)+"s")