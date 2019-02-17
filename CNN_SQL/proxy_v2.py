import socket
import threading
from time import sleep
from test import init,check,convert2label
BUFFSIZE=4096
model,w_model=init()
text_content ='''HTTP/1.x 200 ok
Content-Type: text/html

'''
image_content ='''HTTP/1.x 200 ok
Content-Type: image/jpg

'''
hack=open('hack.jpg','rb')
hacks=image_content.encode()+ hack.read()
hack.close()
def proxy_handler(client_socket,server_ip,server_port,BUFFER ):
	total_data="".encode()
	try:
		server_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		server_socket.connect((server_ip,server_port))
		server_socket.send(BUFFER)
		#server_socket.settimeout(3)
	except:
		print("Something wrong!")
	while True:

		local_buffer=server_socket.recv(BUFFSIZE)
		if not local_buffer:
			break
		else :
			client_socket.sendall(local_buffer)
	#client_socket.send(' '.encode())
	server_socket.close() 
	client_socket.close()
def main():
	local_ip="0.0.0.0"
	local_port=80
	local_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	local_socket.bind((local_ip,local_port))
	local_socket.listen(50)
	poll=[]
	while True:
		#sleep(2)
		print("waiting for cnnection!")
		con,addrs=local_socket.accept()
		poll.append((con,addrs))
		#client.setblocking(0)
		while len(poll):
			(client,addr)=poll.pop()
			buff=client.recv(BUFFSIZE)
		
			data=buff.decode()
			print("conecting from",addr)
			try:
				path=data.split(' ')[1]
			except:
				client.close()
				continue
			if path=='/hack.jpg':
				client.send(hacks)
				client.close()
			else :
				lines=len(data.split('\r\n'))
				i=0
				while(lines>i):
					#print(data.split('\r\n')[i])
					if len(data.split('\r\n')[i]):
						checked=check(model,w_model,data.split('\r\n')[i])
						if(checked==[0]):
							i+=1
						else :
							break
					else:
						i+=1
				if(checked!=0):
					client.send((text_content+convert2label(checked)).encode())
				else:
				
					proxy_thread=threading.Thread(target=proxy_handler,args=(client,"10.10.10.133",80,buff))
					proxy_thread.start()
					print("client close:",addr)
			
if __name__=="__main__":
	main()