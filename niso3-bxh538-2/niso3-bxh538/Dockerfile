FROM pklehre/niso-lab3

ADD solution.py /bin
ADD genetic_algorithm.py /bin

RUN apt-get update
RUN apt-get -y install python

CMD ["-username", "bxh538", "-submission", "python /bin/solution.py"]
