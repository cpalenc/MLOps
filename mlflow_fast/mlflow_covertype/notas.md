## Codigo de installacion de FastApi

pip install "fastapi[all]"

## Codigos en win y linux, en local debe ingresar a la carpeta

python -m uvicorn app:app --reload

python -m uvicorn main:app --reload

python3 -m uvicorn app:app --reload

python3 -m uvicorn main:app --reload

locust -f locustfile.py

## Paginas de verificacion de app

http://127.0.0.1:8000/docs

http://127.0.0.1:8000/redoc

http://localhost:8000/

## Codigos para inspeccionar docker

docker exec -it <nombre-del-contenedor> /bin/bash
docker exec -it fastapi_container /bin/bash
docker exec -it locustmaster /bin/bash
docker exec -it locustworker /bin/bash

## Codigos para borrar lo de docker

docker stop $(docker ps -aq)

docker rm $(docker ps -a -q)

docker container prune

docker image prune -af

docker network rm $(docker network ls -q)

docker system prune --volumes -a


{
  "elevation": 2998,
  "horizontal_distance_to_roadways": 319,
  "hillshade_9am": 233,
  "horizontal_distance_to_fire_points": 955
}

