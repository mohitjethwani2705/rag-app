# TableFlow Integrations and API

## POS integrations
TableFlow connects to Square, Toast and Clover. POS integrations require the Growth or Pro plan.
Square and Clover sync in real time using webhooks. Toast syncs by polling every 15 minutes.
To connect a POS go to Settings > Integrations > Connect. Only an Owner or Manager can do this.

## Reserve with Google
Reserve with Google requires the Growth or Pro plan. Google approval usually takes 5 to 10 business days after you submit.

## Mailchimp
Guest emails sync to Mailchimp every night at 2 AM local time. Only guests who opted in to marketing are synced.

## Zapier
Zapier is available on the Growth and Pro plans.

## API access
The API is available on the Pro plan only. Requests are limited to 100 requests per minute per location.
Authenticate by sending your API key in the X-TableFlow-Key header. Keys can be rotated from Settings > API.

## Webhooks
Webhook events are reservation.created, reservation.cancelled and guest.updated.
A failed webhook delivery is retried up to 5 times with exponential backoff.
