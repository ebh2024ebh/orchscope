# Scanning-campaign replay (offline demo)

A self-contained web page that replays one week of campaign inference at the Merit ORION
network telescope (Sep 24-30, 2026), hour by hour, as the hourly pipeline reported it.

* **No server, no network.** Open `index.html` directly from disk, or copy this folder to any
  static web host. The page loads only its four local files (`index.html`, `style.css`,
  `app.js`, `data.js`).
* **Anonymized.** `data.js` holds campaign-level summaries only: bot counts, country tallies,
  header-field classes, hourly activity, and probe destinations as offsets inside the monitored
  block. It contains no source addresses, ASNs, organizations, or host names.
* **Deep links.** `index.html#c=12&h=40` opens campaign C12 at hour 40 of the window.

`data.js` is produced by `code/export_demo.py` from the week results. `make_sample_data.py`
writes a synthetic stand-in with the same schema for UI development (it shows a "Synthetic
sample data" badge).
