FROM node:21.5-bullseye-slim

WORKDIR /app/web

COPY package.json package.json
RUN npm i

COPY . .

CMD ["npm", "start"]
