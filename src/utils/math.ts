export function computeEffectiveImportance(
  baseImportance: number,
  lastAccessedAt: string | null | undefined,
  createdAt: string,
  floor: number = 2.0,
  decayRate: number = 0.95
): number {
  if (baseImportance <= 0) return 0;
  
  const targetDate = lastAccessedAt || createdAt;
  const daysSince = Math.max(0, (Date.now() - new Date(targetDate).getTime()) / 86400000);
  
  const effective = baseImportance * Math.pow(decayRate, daysSince);
  
  // Apply floor, but don't artificially raise items that were explicitly scored below the floor
  const actualFloor = Math.min(floor, baseImportance);
  return Math.round(Math.max(actualFloor, effective) * 100) / 100;
}
