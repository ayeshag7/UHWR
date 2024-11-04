import socket

pipe = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
pipe.connect(('data.pr4e.org',443))
cmd = "GET http://data.pr4e.org/page1.html \r\n\r\n".encode()
pipe.send(cmd)

while True:
    data = pipe.recv(512)
    if len(data)<1:
        print('connection Error')
        break
    print(data)
    input("PArh le")
pipe.close()