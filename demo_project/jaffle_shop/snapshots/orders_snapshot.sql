{% snapshot orders_snapshot %}

{#-
Normally we would select from the table here, but we are mostly using seeds to load
our data in this demo project
#}

{{
    config(
      target_database='db',
      target_schema='snapshots',
      unique_key='id',

      strategy='check',
      check_cols='all'
    )
}}

select *
from {{ ref('raw_orders') }}

{% endsnapshot %}