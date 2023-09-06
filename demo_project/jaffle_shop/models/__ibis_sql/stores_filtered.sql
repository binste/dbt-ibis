SELECT
  t0.store_id,
  t0.store_name
FROM {{ ref('stg_stores') }} AS t0
WHERE
  t0.store_id = CAST(1 AS TINYINT)