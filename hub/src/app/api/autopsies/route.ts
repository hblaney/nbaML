import { authorizeRequest, jsonNoStore } from "@/lib/auth";

export const dynamic = "force-dynamic";

function detectAlertChannels(): string[] {
  const ch: string[] = [];
  if (
    process.env.TWILIO_ACCOUNT_SID &&
    process.env.TWILIO_AUTH_TOKEN &&
    process.env.TWILIO_FROM_NUMBER &&
    process.env.ALERT_TO_PHONE
  ) {
    ch.push("twilio_sms");
  }
  if (process.env.NTFY_TOPIC) ch.push("ntfy_push");
  if (process.env.PUSHOVER_USER_KEY && process.env.PUSHOVER_API_TOKEN) ch.push("pushover");
  if (process.env.TELEGRAM_BOT_TOKEN && process.env.TELEGRAM_CHAT_ID) ch.push("telegram");
  if (process.env.ALERT_WEBHOOK_URL) ch.push("webhook");
  return ch;
}

export async function GET(request: Request) {
  const caller = await authorizeRequest(request);
  if (caller instanceof Response) return caller;

  const channels = detectAlertChannels();
  return jsonNoStore({
    configured: channels.length > 0,
    channels,
    recentAutopsies: [
      {
        date: "Thu Nov 12, 2026",
        slip: "3-Pick Power ($4 -> $10 Cashout, +150%)",
        legs: [
          {
            player: "Ayo Dosunmu (MIN)",
            pick: "MORE 17.5 PTS",
            actual: "18 PTS (27m)",
            tag: "CLV_SAVED_WIN",
            note: "Beat DK's 18.5 line by 1.0 pt; landed on exact 18 before sitting Q4 in 133-94 blowout.",
          },
          {
            player: "John Collins (DET)",
            pick: "LESS 21.5 PRA",
            actual: "14 PRA (25m, 5 PF)",
            tag: "SURVIVED_OVERTIME_WIN",
            note: "Cleared LESS 21.5 easily even through 2OT; 5 fouls capped extra OT floor time.",
          },
          {
            player: "Jalen Green (PHX)",
            pick: "LESS 20.5 PRA",
            actual: "16 PRA (21m, -16 +/-)",
            tag: "DISCIPLINED_LIVE_CASHOUT",
            note: "Cashed out $10 in 1-pt game; coach kept him benched all Q4 on -16 +/- -> add +/- hook signal to live Q4 model.",
          },
        ],
      },
      {
        date: "Fri Nov 13, 2026 (Completed Early Legs)",
        slip: "5-Pick Flex (In Progress)",
        legs: [
          {
            player: "Landry Shamet (NY)",
            pick: "LESS 11.5 PR",
            actual: "8 PR (21m)",
            tag: "GOOD_PROCESS_WIN",
            note: "+1.0 pt CLV (PP 11.5 vs DK 10.5) + Mikal Bridges return cut spot-starter minutes to 21m.",
          },
          {
            player: "Desmond Bane (ORL)",
            pick: "LESS 20.5 PTS",
            actual: "21 PTS (6/10 3PT, 6 PF)",
            tag: "GOOD_PROCESS_3PT_VARIANCE_LOSS",
            note: "+1.0 pt CLV & fouled out in 35m, but shot 60% on 10 threes (+7.0 pts above expected 3PT). Directly validated RD-001 (-0.51% Points variance penalty).",
          },
        ],
      },
    ],
  });
}
