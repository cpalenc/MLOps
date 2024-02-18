### Build container
docker build -t <container-name> .

### Run image 
docker run --name <container-name_alias>  -p 8000:80 <container-name> 

### List containers
#### active containers 
docker ps
#### all containers
docker ps -a

### remove container
docker rm <id_contaner>
#### remove all stopped containers
docker container  prune

### run ubuntu
docker run ubuntu
#### always up container still running
docker run --name alwaysup -d ubuntu tail -f /dev/null
### conect to cotainer ubuntu and run commands
docker exec -it alwaysup bash
### stop docker
docker stop alwaysup
## Run in backgrond
-d

### volumes
docker volume ls
### create voume
docker volume create <volume_name>