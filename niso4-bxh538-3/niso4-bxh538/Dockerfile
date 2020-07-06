FROM pklehre/niso-lab4

ADD solution.py /bin

RUN apt-get update
RUN apt-get -y install python
RUN apt-get -y install python-numpy

CMD ["-username", "bxh538", "-submission", "python /bin/solution.py"]
