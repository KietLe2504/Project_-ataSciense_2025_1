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
docker run -d --name clickhouse-server --ulimit nofile=262144:262144 -p 8123:8123 -p 9000:9000 -v clickhouse_data:/var/lib/clickhouse -v clickhouse_logs:/var/log/clickhouse-server clickhouse/clickhouse-server
```
- clickhouse-server: name of your container, recommended using this name
- 8123 and 9000: two ports for access, 9000 for terminal and 8123 for web [http://localhost:8123](http://localhost:8123) and Python
- clickhouse_data and clickhouse_logs: volumes for data, remain in Docker even if the container is removed
- clickhouse/clickhouse-server: Clickhouse image pulled above

You can use these commands to check if the container is running and the created volumes:
```bash
docker ps
docker volume ls
```

## Create account in container
Now run this command to access the clickhouse-server bash:
```bash
docker exec -it clickhouse-server bash
```
You are now in the container.

Run this follwing command to access to etc/clickhouse-server:
```bash
cd etc/clickhouse-server
```
Update and install these packages:
```bash
apt update
apt install nano
```
Open users.xml:
```bash
nano users.xml
```
Add this account in user section:
```bash
<admin>
    <password>your-custom-password</password>
    <networks>
        <ip>::/0</ip>
    </networks>
    <profile>default</profile>
    <quota>default</quota>
    <access_management>1</access_management>
</admin>
```
- This is an account with username admin
- There is an account with username default already in the file
- It is recommended to set password for default account and its access_management to 0
- Ctrl + O and press Enter to Save, Ctrl + X to exit

## Set up Clickhouse in Python
Install module clickhouse_connect:
```bash
pip install clickhouse_connect
```
Open python file and insert the following code:
```python
import clickhouse_connect

client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,
    username="admin",
    password="your-custom-password"
)
```

## Notation
Before using Clickhouse on Python, you must start Docker and clickhouse-server:
```bash
docker start clickhouse-server
```
Stop clickhouse-server:
```bash
docker stop clickhouse-server
```
Remove clickhouse-server:
```bash
docker rm -f clickhouse-server
```
You also need need to delete the volumes:
```bash
docker volume rm clickhouse_data
docker volume rm clickhouse_logs
```
