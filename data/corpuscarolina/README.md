# Preparação do Corpus Carolina

O script `prepare.py` baixa o corpus em streaming, tokeniza cada documento e
produz os arquivos:

- `processed/train.bin`
- `processed/val.bin`
- `processed/test.bin`

Cada arquivo é uma sequência de IDs de tokens sem cabeçalho, armazenados como
`uint16` por padrão. O split é determinístico por documento (90% treino, 5%
validação e 5% teste), usando a seed 42.

Para fazer uma execução curta antes de processar o corpus completo:

```bash
uv run python data/corpuscarolina/prepare.py \
  --taxonomies pub \
  --max-documents 20
```

Para preparar todas as taxonomias:

```bash
uv run python data/corpuscarolina/prepare.py
```

O Carolina tem apenas o split original `corpus`; por isso, os três splits são
criados localmente. Use `--seed`, `--val-ratio` e `--test-ratio` para alterar
essa divisão. Para fixar uma versão específica do corpus, passe, por exemplo,
`--revision v2.0.1`.

Os arquivos existentes não são alterados por padrão. Use `--overwrite` para
substituí-los. Consulte `--help` para ver todas as opções.
