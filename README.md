# Setup Clickhouse in Python by using Docker
Clickhouse can allow processing big databases in Python better than traditional methods such as Pandas etc.

This README.md file will introduce how to:
- Install Docker
- Setup Clickhouse in Docker
- Use Clickhouse in Python

## Install and run Docker
Click [here](https://www.docker.com/products/docker-desktop/) to install Docker.

After installing Docker, turn on it and leave it running.

## Pull Clickhouse server image into Docker
On Windows, open Windows PowerShell or Cmd and run this command:
```bash
docker pull clickhouse/clickhouse-server:latest
```

## Create new container for Clickhouse database
Run this command:
```bash
docker run -d --name <container-name> --ulimit nofile=262144:262144 -p 8123:8123 -p 9000:9000 -v clickhouse_data:/var/lib/clickhouse -v clickhouse_logs:/var/log/clickhouse-server clickhouse/clickhouse-server
```
- container-name: name of your container, e.g. clickhouse-server
- 8123 and 9000: two ports for access, 9000 for terminal and 8123 for web [http://localhost:8123](http://localhost:8123) and Python
- clickhouse_data and clickhouse_logs: volumes for data, remain in Docker even if the container is removed
