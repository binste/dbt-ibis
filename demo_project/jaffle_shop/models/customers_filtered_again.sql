-- LICENSE NOTICE: This file was added compared to the original version

select *
from {{ ref('customers_filtered') }}