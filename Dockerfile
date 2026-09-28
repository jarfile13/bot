FROM node:20-slim
WORKDIR /app
COPY bot.js .
EXPOSE 3000
CMD ["node", "bot.js"]
