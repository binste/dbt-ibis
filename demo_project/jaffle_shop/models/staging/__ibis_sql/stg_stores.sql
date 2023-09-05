SELECT
  CAST(t0.store_id AS BIGINT) AS store_id,
  t0.store_name
FROM {{ source('sources_db', 'stores') }} AS t0