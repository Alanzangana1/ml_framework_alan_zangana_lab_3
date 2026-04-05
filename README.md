
# M3 – PyTorch API

## Projektbeskrivning
En tränad SimpleCNN-modell på CIFAR-10 som klassificerar bilder via ett FastAPI REST API och distribueras med Docker.

## Modell
- **Arkitektur:** SimpleCNN
- **Dataset:** CIFAR-10
- **Exportformat:** ONNX
- **Klasser:** plane, car, bird, cat, deer, dog, frog, horse, ship, truck

## Kom igång

### Bygg och starta med Docker
```bash
docker build -t m3-pytorch-api .
docker run -p 8000:8000 m3-pytorch-api
```

### Testa API:et
Gå till http://localhost:8000/docs

## Pull Requests
| # | Titel | Länk |
|---|-------|------|
| PR #1 | Add model architecture and training script | [🔗 Länk](https://github.com/Alanzangana1/ml_framework_alan_zangana_lab_3/pull/1) |
| PR #2 | Add FastAPI endpoint and Dockerfile | [🔗 Länk](https://github.com/Alanzangana1/ml_framework_alan_zangana_lab_3/pull/2) |

## Teknologier
- Python 3.11
- PyTorch
- ONNX Runtime
- FastAPI
- Docker
- uv
```

