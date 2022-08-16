FROM node:14

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

COPY . .
RUN npm install -g serve

EXPOSE 3000
CMD [ "sh", "-c", "serve -s build"]