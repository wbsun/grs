all: rsc rsg node client bwc sc

fec: fec.c fec.h
	gcc -O2 -c fec.c -o fec.o

test: test.c utils.h
	gcc -O2 -c test.c -o test.o

node: node.cc utils.h grskv.h
	g++ -O2 node.cc -o node

rsc: fec test
	gcc -O2 test.o fec.o -o rsc

rsg: rs.cu fec.h fec.c utils.h
	nvcc -O3 -arch=sm_35 rs.cu -o rsg

client: client.cu utils.h fec.h grskv.h fec.c
	nvcc -O3 -arch=sm_35 client.cu -o client

bwc: bwc.cc utils.h grskv.h
	g++ -O2 bwc.cc -o bwc

sc: sc.cu utils.h fec.h grskv.h fec.c
	nvcc -O3 -arch=sm_35 sc.cu -o sc

clean:
	rm -rf *.o
	rm -rf sc
	rm -rf bwc
	rm -rf rsc
	rm -rf rsg
	rm -rf node
	rm -rf client
