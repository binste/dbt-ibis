SELECT
  t0.customer_id,
  t0.first_name,
  t0.last_name,
  t0.first_order,
  t0.most_recent_order,
  t0.number_of_orders,
  t0.customer_lifetime_value
FROM {{ ref('customers') }} AS t0
WHERE
  t0.number_of_orders > CAST(1 AS TINYINT)