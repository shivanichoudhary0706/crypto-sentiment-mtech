# ---------------- live paper trade (frozen models) ----------------
# Paste this whole block into dashboard/app.py just ABOVE the line
#   # ---------------- system health ----------------
# (ASCII only on purpose, so Windows encodings cannot garble it.)
with tab_paper:
    st.warning("Preliminary live operational validation of the FROZEN models. "
               "The sample is far too small for accuracy claims and the three coins move "
               "together; the walk-forward backtest (AUC ~0.53) is the accuracy evidence.")
    cost = st.select_slider("Cost per side (bps)", options=[0, 5, 10, 15, 20], value=10,
                            key="pt_cost")
    pt = get("/paper-trade", freq=freq, cost_bps=cost)
    if not pt or not pt.get("summary"):
        st.info(f"No frozen {freq} signals yet.")
    else:
        s, ops = pt["summary"], pt["operations"]
        st.caption(f"Window {fmt_ts(s['window_start'])} -> {fmt_ts(s['window_end'])} (decision "
                   f"times) | models: {', '.join(ops['model_versions'])}")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("BUY/SELL calls right", f"{s['n_right']} / {s['n_scored']}",
                  help="HOLD is not scored. Outcome = next bar close vs decision close.")
        ci = ("n/a" if s["wilson_lo"] is None
              else f"{s['wilson_lo'] * 100:.0f}-{s['wilson_hi'] * 100:.0f}%")
        k2.metric("Wilson 95% CI of hit rate", ci)
        k3.metric("Independent decision times", s["decision_times"],
                  help=f"right {s['decision_times_right']} / wrong {s['decision_times_wrong']} "
                       f"/ mixed {s['decision_times_mixed']} - all coins at one time ~ one bet")
        up = "n/a" if s["pct_bars_up"] is None else f"{s['pct_bars_up'] * 100:.0f}%"
        k4.metric("Bars up (always-BUY baseline)", up)
        st.caption(f"{s['n_buy']} BUY / {s['n_sell']} SELL / {s['n_hold']} HOLD | "
                   f"pending {s['n_pending']} | missing outcome {s['n_missing']} | "
                   f"suspect (written after outcome, excluded) {s['n_suspect']}")

        st.subheader(f"Paper P&L, long-only, {cost} bps per side")
        eq = pd.DataFrame(pt["equity"])
        if not eq.empty:
            eq["time"] = pd.to_datetime(eq["time"], utc=True)
            fig = go.Figure()
            for sym, g in eq.groupby("symbol"):
                fig.add_trace(go.Scatter(x=g.time, y=(g.paper - 1) * 100, name=f"{short(sym)} paper"))
                fig.add_trace(go.Scatter(x=g.time, y=(g.buy_hold - 1) * 100,
                                         name=f"{short(sym)} buy & hold", line=dict(dash="dot")))
            fig.add_hline(y=0, line_dash="dot", line_color="#8a93a6")
            fig.update_layout(height=380, yaxis_title="Cumulative return %",
                              margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig)
        pnl = pd.DataFrame(pt["pnl"])
        if not pnl.empty:
            pnl["symbol"] = pnl["symbol"].map(short)
            st.dataframe(pnl, hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%.4f")
                for c in ("paper_return", "buy_hold_return", "time_in_market")})

        st.subheader("Operations")
        od = pd.DataFrame(ops["per_symbol"])
        if not od.empty:
            od["symbol"] = od["symbol"].map(short)
            for c in ("first", "last"):
                od[c] = od[c].map(fmt_ts)
            st.dataframe(od, hide_index=True)
        if ops["latency_p50_s"] is not None:
            st.caption(f"Signal latency (bar close -> first DB write, rows after the created_at fix, "
                       f"n={ops['n_latency']}): p50 {ops['latency_p50_s'] / 60:.1f} min, "
                       f"p95 {ops['latency_p95_s'] / 60:.1f} min")

        st.subheader("Every signal")
        sd = pd.DataFrame(pt["signals"])
        sd["valid_from"] = sd["valid_from"].map(fmt_ts)
        sd["symbol"] = sd["symbol"].map(short)
        sd = sd.iloc[::-1]                                   # newest first
        st.dataframe(sd[["valid_from", "symbol", "action", "prob_up", "close", "next_close",
                         "next_ret_bps", "result"]], hide_index=True,
                     column_config={"prob_up": st.column_config.NumberColumn(format="%.3f"),
                                    "next_ret_bps": st.column_config.NumberColumn(
                                        "next bar (bps)", format="%.1f")})

