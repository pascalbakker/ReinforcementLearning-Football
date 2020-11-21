FROM gfootball
WORKDIR /usr/src/app
COPY . .
CMD ["main.py"]
ENTRYPOINT ["python3"]
