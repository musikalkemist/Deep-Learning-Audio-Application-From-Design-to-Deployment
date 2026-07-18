# Pull nginx image from Docker Hub
FROM nginx

# Remove the default nginx config file and replace it with the new one
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/