/** Uppercase without diacritics — correct for Modern Greek (ΑΕΙ not ΆΈΊ). */
export function toGreekUpper(s: string): string {
	return s
		.normalize('NFD')
		.replace(/[̀-ͯ]/g, '')
		.replace(/[΄΅]/g, '')
		.toUpperCase();
}
