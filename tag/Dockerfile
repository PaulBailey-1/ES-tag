FROM public.ecr.aws/docker/library/node:latest AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM public.ecr.aws/docker/library/node:latest AS app
WORKDIR /app
COPY package*.json .
RUN npm ci --only=production
COPY --from=build /app/build ./build

EXPOSE 5000
CMD ["npm", "run", "start"]